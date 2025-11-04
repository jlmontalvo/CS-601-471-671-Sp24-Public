import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn as nn 
from torch.utils.data import DataLoader
from datasets import load_dataset
import evaluate as evaluate
from transformers import get_scheduler
from transformers import AutoModel, AutoModelForSequenceClassification
import argparse
import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm

import time
import os

# Related to BERT
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings

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

class CustomModelforSequenceClassification(nn.Module):

    def __init__(self, model_name, num_labels=2, type="full"):
        super(CustomModelforSequenceClassification, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.type = type
        self.num_labels = num_labels
        # Initialize prefix on CPU; it will be moved to the correct device when model.to(device) is called
        self.prefix = torch.nn.Parameter(torch.randn(prefix_length, self.model.config.hidden_size, requires_grad=True))
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        
        # For head-tuning and prefix-tuning, freeze the base model parameters
        if self.type == "head":
            for param in self.model.parameters():
                param.requires_grad = False
            # Count tuned parameters (only classifier)
            tuned_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
            hidden_size = self.model.config.hidden_size
            print(f"Head-tuning: Only training classifier head")
            print(f"  Hidden size (d): {hidden_size}")
            print(f"  Tuned parameters: {tuned_params} (Formula: 2d + 2 = 2×{hidden_size} + 2 = {2*hidden_size + 2})")
        
        elif self.type == "prefix":
            # Freeze base model parameters for prefix-tuning
            for param in self.model.parameters():
                param.requires_grad = False
            # Count tuned parameters (prefix + classifier)
            prefix_params = self.prefix.numel()
            classifier_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
            tuned_params = prefix_params + classifier_params
            hidden_size = self.model.config.hidden_size
            print(f"Prefix-tuning: Training prefix embeddings and classifier head")
            print(f"  Hidden size (d): {hidden_size}")
            print(f"  Prefix length: {prefix_length}")
            print(f"  Prefix parameters: {prefix_params} (Formula: prefix_length × d = {prefix_length} × {hidden_size})")
            print(f"  Classifier parameters: {classifier_params} (Formula: 2d + 2 = 2×{hidden_size} + 2)")
            print(f"  Total tuned parameters: {tuned_params} (Formula: prefix_length×d + 2d + 2 = {prefix_length}×{hidden_size} + 2×{hidden_size} + 2 = {prefix_length * hidden_size + 2 * hidden_size + 2})")

    def forward(self, input_ids, attention_mask):
        
        if self.type == "full":
            # Full model implementation
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = output.last_hidden_state  # get last hidden state
            mean_hidden = torch.mean(last_hidden, dim=1)  # mean over sequence length
            logits = self.classifier(mean_hidden)  # classify
            return {"logits": logits}

        elif self.type == "head":
            # Head-tuned model (same as full)
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = output.last_hidden_state
            mean_hidden = torch.mean(last_hidden, dim=1)
            logits = self.classifier(mean_hidden)
            return {"logits": logits}
        
        elif self.type == 'prefix':
            # Prefix-tuned model
            batch_size = input_ids.shape[0]
            # Create batch_size copies of prefix
            prefix_repeated = self.prefix.unsqueeze(0).repeat(batch_size, 1, 1)
            
            # Get input embeddings
            input_embeds = self.model.embeddings.word_embeddings(input_ids)
            
            # Concatenate prefix with input embeddings
            inputs_embeds = torch.cat([prefix_repeated, input_embeds], dim=1)
            
            # Move to correct device
            inputs_embeds = inputs_embeds.to(device=input_ids.device)
            
            # Create prefix attention mask (all 1s) and concatenate
            prefix_attention_mask = torch.ones(batch_size, prefix_length, device=attention_mask.device)
            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
            
            # Forward pass with embeddings
            output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            last_hidden = output.last_hidden_state
            mean_hidden = torch.mean(last_hidden, dim=1)
            logits = self.classifier(mean_hidden)
            
            return {"logits": logits}

# In evaluate_model function:
def evaluate_model(model, dataloader, device):
    """
    Evaluate a PyTorch Model
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = output['logits']
            predictions = torch.argmax(predictions, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_references.extend(batch['labels'].numpy())

    # compute and return metrics
    accuracy = accuracy_score(all_references, all_predictions)
    return {'accuracy': accuracy}


def train(mymodel, num_epochs, train_dataloader, validation_dataloader, test_dataloder, device, lr, model_name):
    """ Train a PyTorch Module

    :param torch.nn.Module mymodel: the model to be trained
    :param int num_epochs: number of epochs to train for
    :param torch.utils.data.DataLoader train_dataloader: DataLoader containing training examples
    :param torch.utils.data.DataLoader validation_dataloader: DataLoader containing validation examples
    :param torch.device device: the device that we'll be training on
    :param float lr: learning rate
    :param string model_name: the name of the model
    :return None
    """

    # here, we use the AdamW optimizer. Use torch.optim.AdamW
    print(" >>>>>>>>  Initializing optimizer")
    
    weight_decay = 0.01
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in mymodel.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': weight_decay},
        {'params': [p for n, p in mymodel.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

    # need to customize optimizer for prefix-tuning and head tuning
    custom_optimizer = None

    if mymodel.type == "head":
        # Optimizer for head-tuned model - only train the classifier
        classifier_params = mymodel.classifier.parameters()
        custom_optimizer = torch.optim.AdamW(classifier_params, lr=lr)
    
    elif mymodel.type == "prefix":
        # Optimizer for prefix-tuned model
        # Prefix is a Parameter tensor, need to wrap it properly
        prefix_params = [mymodel.prefix]
        classifier_params = list(mymodel.classifier.parameters())
        custom_optimizer = torch.optim.AdamW(prefix_params + classifier_params, lr=lr)
    

    # now, we set up the learning rate scheduler
    # Use custom_optimizer if available, otherwise use optimizer
    scheduler_optimizer = custom_optimizer if custom_optimizer is not None else optimizer
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=scheduler_optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    loss = torch.nn.CrossEntropyLoss()
    
    epoch_list = []
    train_acc_list = []
    dev_acc_list = []

    for epoch in range(num_epochs):

        epoch_start_time = time.time()

        # put the model in training mode (important that this is done each epoch,
        # since we put the model into eval mode during validation)
        mymodel.train()

        # load metrics - using sklearn for accuracy
        from sklearn.metrics import accuracy_score
        all_train_predictions = []
        all_train_references = []

        print(f"Epoch {epoch + 1} training:")

        for index, batch in tqdm(enumerate(train_dataloader)):

            # get the input_ids, attention_mask, and labels from the batch and put them on the device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # forward pass
            output = mymodel(input_ids=input_ids, attention_mask=attention_mask)
            predictions = output['logits']

            # compute the loss using the loss function
            current_loss = loss(predictions, labels)

            # loss backward
            current_loss.backward()

            # update the model parameters depending on the model type
            if mymodel.type == "full" or mymodel.type == "auto":
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            else:
                custom_optimizer.step()
                lr_scheduler.step()
                custom_optimizer.zero_grad()

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
        
        test_accuracy = evaluate_model(mymodel, test_dataloader, device)
        print(f" - Average test metrics: accuracy={test_accuracy}")

        epoch_end_time = time.time()
        print(f"Epoch {epoch + 1} took {epoch_end_time - epoch_start_time} seconds")

    plot(train_acc_list, dev_acc_list, name=model_name, finetune_method=mymodel.type)

def plot(train_list, valid_list, name, finetune_method):
    
    plt.figure()
    plt.plot(train_list, label='Train')
    plt.plot(valid_list, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.legend()
    plt.savefig(f'{name}_{finetune_method}.png')


def pre_process(model_name, batch_size, device, small_subset, type='auto'):
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
    mytokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

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
    # On Windows, num_workers > 0 can cause issues; use 0 if on Windows, otherwise use multiple workers
    import platform
    num_workers_train = 0 if platform.system() == "Windows" else 4
    num_workers_val = 0 if platform.system() == "Windows" else 2
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=pin_mem, num_workers=num_workers_train)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, pin_memory=pin_mem, num_workers=num_workers_val)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=pin_mem, num_workers=num_workers_val)

    # from Hugging Face (transformers), read their documentation to do this.
    print("Loading the model ...")
    
    if type == "auto":
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    else:
        pretrained_model = CustomModelforSequenceClassification(model_name, num_labels=2, type=type)


    print("Moving model to device ..." + str(device))
    pretrained_model.to(device)
    
    # Verify model is on the correct device
    if hasattr(device, 'type') and device.type == 'cuda':
        model_device = next(pretrained_model.parameters()).device
        device_idx = device.index if device.index is not None else 0
        gpu_name = torch.cuda.get_device_name(device_idx)
        print(f"Model is on device: {model_device}")
        print(f"GPU Name: {gpu_name}")
        
        # Verify it's actually an NVIDIA GPU
        gpu_lower = gpu_name.lower()
        if "intel" in gpu_lower:
            print("⚠⚠⚠ WARNING: Model appears to be on Intel GPU! This will be very slow!")
            print("   Intel GPUs don't support CUDA - this suggests a configuration issue.")
            print("   Try setting CUDA_VISIBLE_DEVICES to hide the Intel GPU.")
        elif "nvidia" in gpu_lower or "geforce" in gpu_lower or "rtx" in gpu_lower or "gtx" in gpu_lower:
            print("✓ Confirmed: Using NVIDIA GPU")
        else:
            print(f"⚠ GPU type unclear from name: {gpu_name}")
        
        if model_device.type != 'cuda':
            print("WARNING: Model is not on CUDA device despite CUDA being available!")
    
    return pretrained_model, train_dataloader, validation_dataloader, test_dataloader


# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_subset", action='store_true')
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None, help="Device to use: 'mps' for Mac, 'cuda' for NVIDIA GPU, 'cpu' for CPU. Auto-detects if not specified.")
    parser.add_argument("--model", type=str, default="roberta-base")
    parser.add_argument("--type", type=str, default="auto", choices=["auto", "full", "head", "prefix"], help="type of tuning to perform on the model")
    parser.add_argument("--prefix_length", type=int, default=128)
    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    assert type(args.small_subset) == bool, "small_subset must be a boolean"
    global prefix_length
    prefix_length = args.prefix_length
    
    # Check for CUDA_VISIBLE_DEVICES environment variable to restrict to NVIDIA GPU
    # If you have Intel + NVIDIA GPUs, you can set this to force NVIDIA
    # Example: set CUDA_VISIBLE_DEVICES=0  (if NVIDIA is device 0)
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible:
        print(f"CUDA_VISIBLE_DEVICES is set to: {cuda_visible}")
    
    # Auto-detect device if not specified
    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
            print("Auto-detected MPS (Mac GPU) device")
        elif torch.cuda.is_available():
            # List all CUDA devices and find NVIDIA GPU
            num_gpus = torch.cuda.device_count()
            print(f"\nFound {num_gpus} CUDA device(s):")
            nvidia_device_idx = None
            device_names = []
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                device_names.append(gpu_name)
                print(f"  Device {i}: {gpu_name}")
                # Check if this is an NVIDIA GPU (explicitly check for NVIDIA keywords)
                gpu_lower = gpu_name.lower()
                # Exclude Intel, AMD, or other non-NVIDIA GPUs
                is_intel = "intel" in gpu_lower
                is_amd = "amd" in gpu_lower or "radeon" in gpu_lower
                is_nvidia = ("nvidia" in gpu_lower or "geforce" in gpu_lower or 
                            "rtx" in gpu_lower or "gtx" in gpu_lower or
                            "tesla" in gpu_lower or "quadro" in gpu_lower)
                
                if is_nvidia and not is_intel and not is_amd:
                    if nvidia_device_idx is None:  # Use first NVIDIA GPU found
                        nvidia_device_idx = i
                        print(f"    → ✓ Selected as primary NVIDIA GPU (device {i})")
                elif is_intel:
                    print(f"    → ✗ Skipping Intel GPU")
            
            if nvidia_device_idx is not None:
                args.device = f"cuda:{nvidia_device_idx}"
                selected_gpu = torch.cuda.get_device_name(nvidia_device_idx)
                print(f"\n✓✓✓ Using NVIDIA GPU: Device {nvidia_device_idx}")
                print(f"   GPU Name: {selected_gpu}")
            else:
                # Check if device 0 is actually NVIDIA (sometimes it is but name check fails)
                device0_name = torch.cuda.get_device_name(0).lower()
                if "intel" in device0_name:
                    print(f"\n⚠⚠⚠ WARNING: Device 0 appears to be Intel GPU: {device0_name}")
                    print("   This may cause poor performance!")
                    # Try to find any device that's not Intel
                    for i in range(num_gpus):
                        if i != 0 and "intel" not in torch.cuda.get_device_name(i).lower():
                            args.device = f"cuda:{i}"
                            print(f"   Trying device {i} instead: {torch.cuda.get_device_name(i)}")
                            break
                    else:
                        args.device = "cuda:0"
                        print(f"   No alternative found, using device 0")
                else:
                    args.device = "cuda:0"
                    print(f"\n⚠ Using device 0: {torch.cuda.get_device_name(0)}")
                    print(f"   (Not explicitly identified as NVIDIA in name)")
        else:
            args.device = "cpu"
            print("Using CPU device")
    
    # convert to a torch.device object for all .to(...) calls
    device = torch.device(args.device)
    print(f"Using torch.device: {device}")
    
    if device.type == "cuda":
        # enable cuDNN autotuner to select best conv algorithms for your hardware
        torch.backends.cudnn.benchmark = True
        print("CUDA available:", torch.cuda.is_available())
        try:
            device_idx = device.index if device.index is not None else 0
            print(f"Using GPU device index: {device_idx}")
            gpu_name = torch.cuda.get_device_name(device_idx)
            print(f"GPU name: {gpu_name}")
            
            # Get CUDA device properties for verification
            props = torch.cuda.get_device_properties(device_idx)
            print(f"GPU Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"Compute Capability: {props.major}.{props.minor}")
            
            # Verify it's NVIDIA (Intel GPUs shouldn't have CUDA support, but just in case)
            gpu_lower = gpu_name.lower()
            if "intel" in gpu_lower:
                print("\n⚠⚠⚠ CRITICAL WARNING: Selected device appears to be Intel GPU!")
                print("   This is unexpected - Intel GPUs don't support CUDA.")
                print("   Please manually specify NVIDIA GPU with --device cuda:1 (or correct index)")
                print("\n   To find your NVIDIA GPU index, run:")
                print("   python -c \"import torch; [print(f'Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]\"")
            elif "nvidia" in gpu_lower or "geforce" in gpu_lower or "rtx" in gpu_lower or "gtx" in gpu_lower:
                print("✓ Verified: Using NVIDIA GPU")
            
            # Set the current device
            torch.cuda.set_device(device_idx)
            print(f"Current CUDA device: {torch.cuda.current_device()}")
        except Exception as e:
            print(f"Warning: Could not get GPU info: {e}")
    
    #load the data and models
    pretrained_model, train_dataloader, validation_dataloader, test_dataloader = pre_process(args.model,
                                                                                             args.batch_size,
                                                                                             device,
                                                                                             args.small_subset,
                                                                                             args.type)
    print(" >>>>>>>>  Starting training ... ")
    train(pretrained_model, args.num_epochs, train_dataloader, validation_dataloader, test_dataloader, device, args.lr, args.model)
    
    # print the GPU memory usage just to make sure things are alright
    print_gpu_memory()

    val_accuracy = evaluate_model(pretrained_model, validation_dataloader, device)
    print(f" - Average DEV metrics: accuracy={val_accuracy}")

    test_accuracy = evaluate_model(pretrained_model, test_dataloader, device)
    print(f" - Average TEST metrics: accuracy={test_accuracy}")
