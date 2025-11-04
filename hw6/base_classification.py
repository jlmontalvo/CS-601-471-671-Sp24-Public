import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import evaluate as evaluate
from transformers import get_scheduler
from transformers import AutoModelForSequenceClassification
import argparse
import subprocess
import matplotlib.pyplot as plt

def print_gpu_memory():
    """
    Print the amount of GPU memory used by the current process
    This is useful for debugging memory issues on the GPU
    """
    # check if MPS (Mac) is available
    if torch.backends.mps.is_available():
        print("Using Metal Performance Shaders (MPS) on Mac")
        # MPS doesn't have the same memory reporting as CUDA
        print("MPS memory reporting is limited - using device for acceleration")
    # check if CUDA is available (for Linux/Windows)
    elif torch.cuda.is_available():
        print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
        try:
            p = subprocess.check_output('nvidia-smi')
            print(p.decode("utf-8"))
        except:
            pass  # nvidia-smi not available
    else:
        print("Using CPU - no GPU acceleration available")


class BoolQADataset(torch.utils.data.Dataset):
    """
    Dataset for the dataset of BoolQ questions and answers
    """

    def __init__(self, passages, questions, answers, tokenizer, max_len):
        self.passages = passages
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, index):
        """
        This function is called by the DataLoader to get an instance of the data
        :param index:
        :return:
        """

        passage = str(self.passages[index])
        question = self.questions[index]
        answer = self.answers[index]

        # this is input encoding for your model. Note, question comes first since we are doing question answering
        # and we don't wnt it to be truncated if the passage is too long
        input_encoding = question + " [SEP] " + passage

        # encode_plus will encode the input and return a dictionary of tensors
        encoded_review = self.tokenizer.encode_plus(
            input_encoding,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        return {
            'input_ids': encoded_review['input_ids'][0],  # we only have one example in the batch
            'attention_mask': encoded_review['attention_mask'][0],
            # attention mask tells the model where tokens are padding
            'labels': torch.tensor(answer, dtype=torch.long)  # labels are the answers (yes/no)
        }


def evaluate_model(model, dataloader, device):
    """ Evaluate a PyTorch Model
    :param torch.nn.Module model: the model to be evaluated
    :param torch.utils.data.DataLoader test_dataloader: DataLoader containing testing examples
    :param torch.device device: the device that we'll be training on
    :return accuracy
    """

    # load metrics - using sklearn accuracy as fallback since evaluate library has issues
    from sklearn.metrics import accuracy_score
    all_predictions = []
    all_references = []

    # turn model into evaluation mode
    model.eval()

    # iterate over the dataloader
    with torch.no_grad():
        for batch in dataloader:
            # get the input_ids, attention_mask from the batch and put them on the device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # forward pass
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            
            predictions = output.logits
            predictions = torch.argmax(predictions, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_references.extend(batch['labels'].numpy())

    # compute and return metrics
    accuracy = accuracy_score(all_references, all_predictions)
    return {'accuracy': accuracy}


def train(mymodel, num_epochs, train_dataloader, validation_dataloader, test_dataloder, device, lr, small_subset=False):
    """ Train a PyTorch Module

    :param torch.nn.Module mymodel: the model to be trained
    :param int num_epochs: number of epochs to train for
    :param torch.utils.data.DataLoader train_dataloader: DataLoader containing training examples
    :param torch.utils.data.DataLoader validation_dataloader: DataLoader containing validation examples
    :param torch.device device: the device that we'll be training on
    :param float lr: learning rate
    :return None
    """

    # here, we use the AdamW optimizer. Use torch.optim.Adam.
    # instantiate it on the untrained model parameters with a learning rate of 5e-5
    print(" >>>>>>>>  Initializing optimizer")
    optimizer = torch.optim.AdamW(mymodel.parameters(), lr=lr)

    # now, we set up the learning rate scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    loss = torch.nn.CrossEntropyLoss()
    
    epoch_list = []
    train_acc_list = []
    dev_acc_list = []

    for epoch in range(num_epochs):

        # put the model in training mode (important that this is done each epoch,
        # since we put the model into eval mode during validation)
        mymodel.train()

        # load metrics - using sklearn for accuracy
        from sklearn.metrics import accuracy_score
        all_train_predictions = []
        all_train_references = []

        print(f"Epoch {epoch + 1} training:")

        for i, batch in tqdm(enumerate(train_dataloader)):
            # get the input_ids, attention_mask, and labels from the batch and put them on the device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # forward pass
            output = mymodel(input_ids=input_ids, attention_mask=attention_mask)
            predictions = output.logits

            # compute the loss using the loss function
            loss_value = loss(predictions, labels)

            # loss backward
            loss_value.backward()

            # update the model parameters with optimizer and lr_scheduler step
            optimizer.step()
            lr_scheduler.step()
            
            # zero the gradients
            optimizer.zero_grad()

            # compute predictions for accuracy
            pred_labels = torch.argmax(predictions, dim=1)
            all_train_predictions.extend(pred_labels.cpu().numpy())
            all_train_references.extend(batch['labels'].numpy())
            
        # print evaluation metrics
        print(f" ===> Epoch {epoch + 1}")
        train_acc = accuracy_score(all_train_references, all_train_predictions)
        print(f" - Average training metrics: accuracy={train_acc}")
        train_acc_list.append(train_acc)

        # normally, validation would be more useful when training for many epochs
        val_accuracy = evaluate_model(mymodel, validation_dataloader, device)
        print(f" - Average validation metrics: accuracy={val_accuracy}")
        dev_acc_list.append(val_accuracy['accuracy'])
        
        epoch_list.append(epoch)
        
    # generate plots here
    plt.clf()
    plt.plot(epoch_list, train_acc_list, 'b', label='train')
    if not small_subset:
        plt.plot(epoch_list, dev_acc_list, 'g', label='valid')
    plt.xlabel('Training Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    save_path = "overfit.png" if small_subset else "base_full.png"
    plt.savefig(save_path)

def pre_process(model_name, batch_size, device, small_subset):
    # download dataset
    print("Loading the dataset ...")
    dataset = load_dataset("boolq")
    dataset = dataset.shuffle()  # shuffle the data

    print("Slicing the data...")
    if small_subset:
        # use this tiny subset for debugging the implementation
        dataset_train_subset = dataset['train'][:10]
        dataset_dev_subset = dataset['train'][:10]
        dataset_test_subset = dataset['train'][:10]
    else:
        # since the dataset does not come with any validation data,
        # split the training data into "train" and "dev"
        dataset_train_subset = dataset['train'][:8000]
        dataset_dev_subset = dataset['validation']
        dataset_test_subset = dataset['train'][8000:]

    print("Size of the loaded dataset:")
    print(f" - train: {len(dataset_train_subset['passage'])}")
    print(f" - dev: {len(dataset_dev_subset['passage'])}")
    print(f" - test: {len(dataset_test_subset['passage'])}")

    # maximum length of the input; any input longer than this will be truncated
    # we had to do some pre-processing on the data to figure what is the length of most instances in the dataset
    max_len = 128

    print("Loading the tokenizer...")
    mytokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loding the data into DS...")
    train_dataset = BoolQADataset(
        passages=list(dataset_train_subset['passage']),
        questions=list(dataset_train_subset['question']),
        answers=list(dataset_train_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )
    validation_dataset = BoolQADataset(
        passages=list(dataset_dev_subset['passage']),
        questions=list(dataset_dev_subset['question']),
        answers=list(dataset_dev_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )
    test_dataset = BoolQADataset(
        passages=list(dataset_test_subset['passage']),
        questions=list(dataset_test_subset['question']),
        answers=list(dataset_test_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )

    print(" >>>>>>>> Initializing the data loaders ... ")
    # use pinned memory and multiple workers when using CUDA for better throughput
    pin_mem = (device.type == "cuda") if hasattr(device, "type") else (str(device) == "cuda")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=pin_mem, num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, pin_memory=pin_mem, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=pin_mem, num_workers=2)

    # from Hugging Face (transformers), read their documentation to do this.
    print("Loading the model ...")
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    print("Moving model to device ..." + str(device))
    pretrained_model.to(device)
    return pretrained_model, train_dataloader, validation_dataloader, test_dataloader


# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_subset", action='store_true',
                        help="When set true, only run training on a small subset of the data, used for 3.1.1")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None, help="Device to use: 'mps' for Mac, 'cuda' for NVIDIA GPU, 'cpu' for CPU. Auto-detects if not specified.")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")

    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    assert type(args.small_subset) == bool, "small_subset must be a boolean"

    # Auto-detect device if not specified
    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
            print("Auto-detected MPS (Mac GPU) device")
        elif torch.cuda.is_available():
            args.device = "cuda"
            print("Auto-detected CUDA device")
        else:
            args.device = "cpu"
            print("Using CPU device")
    # convert to a torch.device object for all .to(...) calls
    device = torch.device(args.device)
    print("Using torch.device:", device)
    if device.type == "cuda":
        # enable cuDNN autotuner to select best conv algorithms for your hardware
        torch.backends.cudnn.benchmark = True
        print("CUDA available:", torch.cuda.is_available())
        try:
            print("GPU name:", torch.cuda.get_device_name(0))
        except Exception:
            pass

    # load the data and models (pass torch.device)
    pretrained_model, train_dataloader, validation_dataloader, test_dataloader = pre_process(
        args.model, args.batch_size, device, args.small_subset)
    print(" >>>>>>>>  Starting training ... ")
    train(pretrained_model, args.num_epochs, train_dataloader, validation_dataloader, test_dataloader, device, args.lr, args.small_subset)
    
    # print the GPU memory usage just to make sure things are alright
    print_gpu_memory()

    val_accuracy = evaluate_model(pretrained_model, validation_dataloader, device)
    print(f" - Average DEV metrics: accuracy={val_accuracy}")

    test_accuracy = evaluate_model(pretrained_model, test_dataloader, device)
    print(f" - Average TEST metrics: accuracy={test_accuracy}")
