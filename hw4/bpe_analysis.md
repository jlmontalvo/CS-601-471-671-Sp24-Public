# BPE Subword Analysis on Wikipedia Text

## Examples of Subword Types

### 1. Complete Words
Examples from the output:
- `development</w>`, `government</w>`, `published</w>`, `American</w>`, `British</w>`, `English</w>`, `French</w>`, `German</w>`, `Portuguese</w>`, `Japanese</w>`, `Australian</w>`, `European</w>`, `United</w>`, `National</w>`, `World</w>`, `War</w>`, `President</w>`, `General</w>`, `Company</w>`, `November</w>`, `December</w>`, `September</w>`, `October</w>`, `January</w>`, `million</w>`, `century</w>`, `original</w>`, `established</w>`, `followed</w>`, `headquarters</w>`, `operations</w>`, `campaign</w>`, `season</w>`, `tropical</w>`, `condoms</w>`, `Thunderbirds</w>`, `Missouri</w>`, `starling</w>`, `birds</w>`

### 2. Part of a Word
Examples from the output:
- `ing`, `tion`, `ation`, `ment`, `ness`, `ity`, `ous`, `ing</w>`, `tion</w>`, `ation</w>`, `ed</w>`, `er</w>`, `ly</w>`, `al</w>`
- Prefixes: `inter`, `pre`, `pro`, `sub`, `con`, `dis`, `un`, `re`
- Word fragments: `dev` (develop), `produc` (production), `tional` (national/international), `ical` (political/medical), `ically` (historically), `ered` (covered/discovered), `ated` (created/located)
- Common endings: `ern</w>` (western/northern), `ish</w>` (English/British), `ies</w>` (stories/eries), `ents</w>` (events/elements)

### 3. Non-English Words and Special Characters
Examples from the output:
- **Georgian script**: `ა`, `ბ`, `გ`, `დ`, `ე`, `ვ`, `ზ`, `თ`, `ი`, `კ`, `ლ`, `მ`, `ნ`, `ო`, `პ`, `ჟ`, `რ`, `ს`, `ტ`, `უ`, `ფ`, `ქ`, `ღ`, `ყ`, `შ`, `ჩ`, `ც`, `ძ`, `წ`, `ჭ`, `ხ`, `ჯ`, `ჰ`, `Ⴀ`, `Ⴂ`, `Ⴃ`, `Ⴈ`, `Ⴌ`, `Ⴕ`, `Ⴟ`, `ჱ`, `ჲ`, `ჳ`, `ჴ`, `ჵ`, `ჶ`, `ჷ`, `ჸ`, `ჹ`, `ჺ`, `჻`
- **Arabic script**: `ا`, `ب`, `ت`, `ش`, `ع`, `ل`, `م`, `و`, `ي`, `د`, `س`, `ن`, `خ`, `ﻋ`
- **Devanagari script**: `अ`, `आ`, `इ`, `ई`, `उ`, `ऊ`, `ए`, `ऐ`, `ओ`, `औ`, `क`, `ख`, `ग`, `घ`, `च`, `छ`, `ज`, `झ`, `ट`, `ठ`, `ड`, `ढ`, `ण`, `त`, `थ`, `द`, `ध`, `न`, `प`, `फ`, `ब`, `भ`, `म`, `य`, `र`, `ल`, `व`, `श`, `ष`, `स`, `ह`, `ा`, `ि`, `ी`, `ु`, `ू`, `ृ`, `े`, `ै`, `ो`, `ौ`, `्`, `ं`, `ः`, `़`
- **Greek letters**: `α`, `β`, `γ`, `δ`, `ε`, `η`, `θ`, `κ`, `λ`, `μ`, `ν`, `ο`, `π`, `ρ`, `ς`, `σ`, `τ`, `υ`, `φ`, `Χ`, `Φ`, `Δ`, `ό`, `ή`, `ώ`, `ὀ`
- **Cyrillic letters**: `а`, `в`, `е`, `к`, `н`, `о`, `р`, `с`, `т`, `у`, `я`, `з`, `и`, `й`, `л`, `С`, `Б`, `Р`, `У`
- **Special characters**: `@-@</w>`, `@.@</w>`, `@,@</w>`, `–</w>`, `—</w>`, `±`, `×`, `÷`, `°</w>`, `″`, `′`, `☉`, `→`, `…`, `·`, `~`, `^`, `#`, `%</w>`, `$</w>`, `£`, `€`, `¢`, `&`, `*`, `+</w>`, `-`, `/</w>`, `=</w>`, `<`, `>`, `[</w>`, `]</w>`, `(</w>`, `)</w>`, `{`, `}`, `"</w>`, `'</w>`, `'`, `'`, `"`, `"`, `;</w>`, `:</w>`, `,</w>`, `.</w>`, `?`, `!`, `'`, `ʃ`, `ə`, `ɛ`, `ɔ`, `ɑ`, `ʊ`, `ɳ`, `ɡ`, `ʁ`, `ʋ`, `ʕ`, `ɢ`, `ː`, `ˈ`, `ˌ`, `ᵻ`
- **Japanese characters**: `動`, `機`, `転`, `攻`, `裁`, `隊`, `殻`, `逆`, `判`
- **Vietnamese diacritics**: `á`, `à`, `ả`, `ã`, `ạ`, `ă`, `ắ`, `ằ`, `ẳ`, `ẵ`, `ặ`, `â`, `ấ`, `ầ`, `ẩ`, `ẫ`, `ậ`, `é`, `è`, `ẻ`, `ẽ`, `ẹ`, `ê`, `ế`, `ề`, `ể`, `ễ`, `ệ`, `í`, `ì`, `ỉ`, `ĩ`, `ị`, `ó`, `ò`, `ỏ`, `õ`, `ọ`, `ô`, `ố`, `ồ`, `ổ`, `ỗ`, `ộ`, `ơ`, `ớ`, `ờ`, `ở`, `ỡ`, `ợ`, `ú`, `ù`, `ủ`, `ũ`, `ụ`, `ư`, `ứ`, `ừ`, `ử`, `ữ`, `ự`, `ý`, `ỳ`, `ỷ`, `ỹ`, `ỵ`, `đ`
- **Language-specific words**: `Bolívar</w>`, `García</w>`, `Márquez</w>`, `Khandoba</w>`, `anekān` (Sanskrit), `tavāda</w>` (Sanskrit), `āda</w>` (Sanskrit), `Varanasi</w>`, `Croati` (Croatian), `ekān` (Sanskrit)
- **Proper names with diacritics**: `Iguanodon</w>`, `Corythosaurus</w>`, `Alkan</w>`, `Mosley</w>`, `Walpole</w>`, `Nixon</w>`, `Anderson</w>`, `Morrison</w>`, `Perry</w>`, `Midge</w>`

## Explanation: Why These Subwords Emerged

These subwords emerged from the BPE algorithm through a data-driven process that repeatedly merges the most frequent character pairs. Complete words like "government" and "American" appear frequently in Wikipedia articles about politics and geography, so the algorithm merged all their characters into single tokens. Word parts like "ing", "tion", and "ment" became subwords because they appear as common suffixes across many different words - it's more efficient to store these as reusable pieces rather than full words. The presence of non-English characters reflects Wikipedia's multilingual nature: Georgian, Arabic, Devanagari, Greek, Cyrillic, Japanese, and Vietnamese scripts appear in articles about those regions or topics. Special characters like "@-@" (hyphen encoding), mathematical symbols (×, ±), and IPA phonetic symbols emerged because Wikipedia articles contain dates, measurements, pronunciations, and scientific notation. BPE naturally discovers these patterns without language-specific rules, creating a vocabulary that balances frequency (common words), morphology (affixes), and coverage (rare scripts and symbols).


