# MTG AI Deck Builder
_Unleash the power of **Machine Learning** to forge next-level **Magic: The Gathering** decks that adapt to the ever-evolving **Commander** meta!_

[![Magic: The Gathering](https://img.shields.io/badge/Magic%3A%20the%20Gathering-AI%20Deck%20Builder-blue)](#)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](#)
[![Unsupervised Learning](https://img.shields.io/badge/Machine%20Learning-Unsupervised-green)](#)

## Overview
**MTG AI Deck Builder** aims to train a **dynamic**, unsupervised **AI model** that consistently updates itself to generate competitive **Commander** decks (with optional support for other formats). By tapping into the **Scryfall** API for card data, it analyzes existing decks and the broader meta—ultimately discovering creative synergies and archetype strategies that might be overlooked by the community.

### Why This Project?
- **Adaptive Metagame Analysis**: Stay ahead of format shifts by re-training on newly released sets or emergent archetypes.
- **Unsupervised Creativity**: Rely on an unsupervised learning framework to find unexpected or "under-the-radar" combinations.
- **Data-Driven**: Harness card data from Scryfall, letting the model continuously refine deck lists as the meta evolves.
- **Solo & Open-Source**: This is primarily a solo project, but feel free to fork and experiment on your own!

## Current Status
> :warning: **Work in progress**: The project features multiple analysis approaches with varying degrees of sophistication.

- **[`fetch_standard_legal_cards.py`]**  
  Despite the legacy filename, this script now fetches cards for any format (default: Commander) from the Scryfall API and outputs them as a CSV dataset. It handles various card layouts—including split cards and Room cards—and extracts comprehensive card data such as types, colors, keywords, and mechanics. Switch formats by passing `--format <format-name>`.

- **[`deck_analysis.py`]**  
  Analyzes an input deck list to produce baseline archetype insights and Commander compliance checks. In addition to archetype tagging (Aggro, Midrange, Control, Tempo, Combo), it now validates commander color identity, singleton rules, ramp density, companion usage, and deck-size requirements while still detailing mana curve, card compositions, and mechanic distributions.

- **[`current_standard_deck_list_scraper.py`]**  
  Scrapes the latest Commander (or other format) deck lists from MTGGoldfish's metagame page. Allows filtering by minimum meta percentage, saves deck lists (with preserved Commander/Deck headers) as text files for analysis, and exports meta representation data as JSON. Use `--format commander` (default) or swap formats as needed.

- **[`analyze_meta_old_try_to_parse.py`]**  
  The first meta analysis script that uses rule-based pattern matching to identify mechanics and synergies. It parses oracle text using predefined regex patterns to detect card interactions, with special handling for Room-type enchantments. While comprehensive, its accuracy depends on the quality of the predefined patterns. It can misclassify complex interactions or miss new mechanics that don't match its patterns.

- **[`analyze_meta_using_keywords.py`]**  
  A statistical approach that analyzes card data directly without assumptions about interactions. It dynamically extracts types, subtypes, keywords, and references from the card pool to identify patterns. This approach offers more reliable results when the card pool changes, as it doesn't rely on hardcoded patterns. It excels at providing objective meta statistics but offers less insight into complex card interactions.

- **[`semantics_meta_analysis.py`]**  
  The newest script that implements machine learning and semantic analysis. It uses a pre-trained sentence transformer model to generate embeddings for cards based on their oracle text. It then applies clustering techniques to identify similar cards and decks without relying on predefined patterns. This approach can discover nuanced relationships and emergent themes that might be missed by rule-based systems, though the identified similarities may sometimes lack clear explanation since the model isn't specifically trained on Magic terminology.

- **[`integrated_deck_name_analyzer.py`]**  
  A sophisticated module that enhances meta analysis by extracting meaningful information from deck names. It dynamically identifies color identities, archetypes, card types, subtypes, and mechanics referenced in deck names without relying on hardcoded patterns. The analyzer can:
  - Extract and normalize card-related terminology from deck names
  - Process compound terms (like "self-bounce") as semantic units
  - Match terms against actual cards in the deck for more precise analysis
  - Perform cross-deck analysis to identify shared themes and synergies
  - Enhance archetype detection by incorporating deck name insights
  - Use TF-IDF to extract relevant terms from oracle texts of common cards
  This analysis adds another layer of insight to understand how the community names and categorizes decks, revealing conceptual connections between different archetypes and strategies.

- **[`consolidated_meta_analysis.py`]**  
  Combines the outputs from all three meta analysis approaches (pattern-based, keyword-based, and semantic) to generate a comprehensive meta report. It reconciles potentially conflicting information from different analysis methods, extracts the most reliable insights from each, and produces a unified view of the metagame including archetype distributions, card type trends, color combinations, and synergy clusters.

- **[`ai_deck_generator.py`]**  
  A first-pass, meta-driven **deck generator**. It uses the same card database and scraped decklists as the analysis scripts, building simple frequency statistics over the existing meta. Given a requested format, color identity, and optional archetype hint, it produces a complete decklist by:
  - Filtering the card pool by color identity and basic Commander/constructed rules  
  - Favoring cards that appear more often in the current meta  
  - Enforcing approximate land ratios and copy limits (Commander singleton vs. 4-of)  
  This module is intentionally simple and model-agnostic so it can later be upgraded to use neural generators and semantic embeddings while keeping the same `DeckSpec` interface.

- **[`consolidated_meta_analysis.py`]**  
  Combines the outputs from all three meta analysis approaches (pattern-based, keyword-based, and semantic) to generate a comprehensive meta report. It reconciles potentially conflicting information from different analysis methods, extracts the most reliable insights from each, and produces a unified view of the metagame including archetype distributions, card type trends, color combinations, and synergy clusters.

All meta analysis scripts are maintained in the repository as they provide complementary insights for different purposes. Use the pattern-matching approach for detailed mechanic breakdowns, the keyword-based approach for reliable statistical analysis, the semantic approach for discovering unexpected card relationships, and the deck name analyzer for understanding deck conceptualization and community categorization.

## Key Features
1. **Scryfall Integration**  
   Automatically pulls the latest **Commander**-legal cards (or any supported format via `--format`), ensuring the model is always up-to-date.
2. **Deck Archetype Analysis**  
   Categorizes decks into established archetypes (Aggro, Midrange, Control, Tempo, Combo) using multiple approaches:
   - Statistical analysis of card distributions
   - Pattern matching on card mechanics
   - Semantic similarity clustering
   - Deck name terminology analysis
3. **Meta Analysis**  
   Multiple approaches to analyze the metagame:
   - Pattern-based mechanic and synergy detection
   - Statistical keyword and type analysis
   - Machine learning-based semantic analysis
   - Community naming convention analysis
4. **Unsupervised Learning Potential**  
   Plans to integrate an AI model that **auto-generates** decklists—unconstrained by conventional archetype thinking.

## Installation
Install dependencies from the provided requirements files:
- `requirements.txt` for runtime dependencies
- `requirements-dev.txt` for runtime + development tooling (tests)

Clone the repo:
```bash
git clone https://github.com/georgejieh/mtg_ai_deck_builder.git
cd mtg_ai_deck_builder
```
Then install dependencies:
```bash
pip install -r requirements.txt
```

For development/test work:
```bash
pip install -r requirements-dev.txt
```

## Local UI
Run the deck generator UI locally with Streamlit:

```bash
pip install -r requirements-ui.txt
streamlit run mtg_ui.py
```

The app opens at `http://localhost:8501`.
The UI now includes:
- `Generate Deck` tab for deck creation
- `Train Model` tab for corpus build + model training

### Docker UI
Build and run with Docker Compose:

```bash
docker compose up --build
```

Then open `http://localhost:8501`.

If you want LLM reranking in the UI, pass your API key:

```bash
OPENAI_API_KEY=your_key_here docker compose up --build
```

## Usage Example

### 1. Fetch Commander-Legal Cards
Pull down all Commander-legal cards (saves `commander_cards.csv` to `./data` by default):
```bash
python fetch_standard_legal_cards.py --format commander
```

### 2. Scrape Current Meta Decks
Scrape the latest Commander deck lists from MTGGoldfish (and keep the Commander/Deck section headers intact):
```bash
python current_standard_deck_list_scraper.py --format commander --min-meta 1.0
```

Alternative source (Archidekt API, useful when MTGGoldfish scraping is flaky or blocked):
```bash
python archidekt_deck_list_scraper.py \
  --format commander \
  --min-meta 0.2 \
  --top-k 500 \
  --max-decks 500 \
  --max-pages 9
```
The Archidekt scraper applies format-aware size filters by default (`Commander: 95-120 cards`) to reduce malformed lists. Override with `--min-total-cards` / `--max-total-cards`.

### 3. Analyze a Deck
To analyze a single deck list, place your deck in a `.txt` file (Commander sections supported) and run:
```bash
python deck_analysis.py /path/to/decklist.txt --cards data/commander_cards.csv
```

### 4. Analyze the Meta
You can use any of the meta analysis scripts based on your needs:
```bash
# For rule-based pattern matching (comprehensive but potentially less accurate):
python analyze_meta_old_try_to_parse.py --cards data/commander_cards.csv --decks current_commander_decks

# For statistical keyword-based analysis (more reliable but less insightful):
python analyze_meta_using_keywords.py --cards data/commander_cards.csv --decks current_commander_decks

# For semantic analysis using machine learning (discovers nuanced relationships):
python semantics_meta_analysis.py --cards data/commander_cards.csv --decks current_commander_decks

# For enhanced semantic analysis with deck name analysis:
python integrated_deck_name_analyzer.py --cards data/commander_cards.csv --decks current_commander_decks
```

### 5. Generate Consolidated Meta Report
Combine insights from all meta analysis approaches:
```bash
python consolidated_meta_analysis.py
```

Each script will generate its own analysis output file and display a summary report in the console.

### 6. Build an Efficient Deck Corpus and Train Clusters

For larger datasets, build the training corpus using a sparse CSR matrix (more memory-efficient than dense arrays):

```bash
python deck_corpus_builder.py \
  --cards data/commander_cards.csv \
  --decks current_commander_decks \
  --output-prefix data/deck_corpus \
  --min-card-frequency 2
```

Then train a lightweight cluster model:

```bash
python train_deck_generator.py \
  --corpus-prefix data/deck_corpus \
  --cards data/commander_cards.csv \
  --semantic-dim 64 \
  --clusters 16 \
  --output-prefix models/deck_kmeans
```

This now trains a hybrid model:
- structure from deck co-occurrence clusters
- semantics from oracle textbox embeddings (TF-IDF + SVD)

The corpus metadata (`<prefix>_meta.json`) includes quality stats such as coverage ratio, unknown-card count, and matrix density.

### 7. Generate a Deck with the Baseline AI Generator

Once you have a card database and a directory of example decklists (Commander or another format), you can ask the generator to produce a new list:

```bash
# Example: generate a Boros (W/R) Commander-style deck
python ai_deck_generator.py \
  --cards data/commander_cards.csv \
  --training-decks current_commander_decks \
  --cluster-model models/deck_kmeans \
  --format commander \
  --colors WR \
  --size 100 \
  --semantic-strength 1.0 \
  --llm-rerank-top-k 20 \
  --llm-strength 0.8 \
  --output generated_decks/boros_commander_ai.txt
```

For non-Commander formats (e.g., Standard), point `--training-decks` at `current_standard_decks` and adjust `--format`/`--size` as needed:

```bash
python ai_deck_generator.py \
  --cards data/commander_cards.csv \
  --training-decks current_standard_decks \
  --format standard \
  --colors WR \
  --size 60 \
  --output generated_decks/boros_standard_ai.txt
```

Use `--include` and `--exclude` to force or ban specific cards, and `--seed` for reproducible outputs. The current generator combines frequency, cluster profile, and optional textbox semantics while keeping the same CLI.

To enable LLM reranking, set an API key in the environment (defaults to `OPENAI_API_KEY`):

```bash
export OPENAI_API_KEY=...
```

Then use `--llm-rerank-top-k` to rerank only the strongest candidates from the fast model. This keeps latency/cost bounded while adding strategic reasoning on top.

#### Decklist Format
Commander decklists downloaded from MTGGoldfish (and parsed by `deck_analysis.py`) typically use explicit section headers instead of blank lines. Example:

<details>
<summary>Sample Deck List</summary>

```
Commander
1 Atraxa, Praetors' Voice

Deck
1 Sol Ring
1 Arcane Signet
1 Cultivate
1 Kodama's Reach
1 Farseek
1 Three Visits
1 Smothering Tithe
1 Farewell
... (remaining 92 cards)
```

</details>

If you're analyzing a non-Commander list, you can still split the mainboard and sideboard with a blank line—the parser will continue to recognize both layouts.

## Comparison of Meta Analysis Approaches

| Feature | Pattern-Based | Keyword-Based | Semantic Analysis | Deck Name Analysis |
|---------|--------------|---------------|-------------------|-------------------|
| **Approach** | Rule-based with regex patterns | Statistical analysis of card data | Machine learning with text embeddings | NLP-based terminology extraction |
| **Strengths** | Detailed mechanic breakdown<br>Synergy identification<br>Room card handling | Reliable with changing card pools<br>Objective meta statistics<br>No assumptions needed | Discovers nuanced relationships<br>Finds emergent themes<br>Not limited by predefined patterns | Reveals community categorization<br>Cross-deck theme identification<br>Enhances archetype detection |
| **Limitations** | May miss new mechanics<br>Pattern accuracy depends on rules<br>Less adaptable | Less insight into card interactions<br>Limited synergy detection<br>More descriptive than analytical | Less interpretable results<br>Model not trained on Magic terminology<br>Requires additional dependencies | Depends on naming conventions<br>May miss concepts not in names<br>Requires accurate card data matching |
| **Best For** | Mechanic & synergy analysis<br>Room card interactions<br>Detailed breakdown | Objective meta statistics<br>Format speed analysis<br>Reliable archetype detection | Discovering unexpected relationships<br>Deck clustering<br>Finding hidden patterns | Understanding meta conceptualization<br>Cross-archetype connections<br>Community terminology analysis |

## Roadmap
- **Enhance Archetype Logic**  
  Incorporate more nuanced synergy/keyword detection as new sets release.
- **Automated Meta Updates**  
  Improve scripts to detect new mechanics automatically, providing real-time meta insights.
- **Neural Network Model**  
  Implement an unsupervised (possibly semi-supervised) approach to **auto-generate** innovative decklists.
- **Self-Training**  
  Continually retrain as new sets and meta changes arise, refining synergy detection beyond current methods.
- **User Interface**  
  Explore a simple web-based front-end for deck analysis and meta breakdown.
- **Train Custom Embeddings**  
  Develop Magic-specific word embeddings to improve the semantic analysis accuracy.
- **Expand Deck Name Analysis**  
  Further refine deck name parsing to extract strategy nuances and detect emerging terminology.

## License
This project is available under the [MIT License](LICENSE). Since this is a solo project, no external contributions are expected—but feel free to fork and experiment.

---

Stay tuned for continuous updates as the **AI Deck Builder** project evolves—pushing the boundaries of MTG tech, one set at a time!
