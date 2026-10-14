# Data

This project runs against the **King County House Sales** dataset (home
sales in the Seattle metro area, King County, WA, 2014-2015).

It's a good fit for this pipeline specifically because it isn't clean: it
has enough missing values and genuine outliers (in `price` and `sqft_lot`)
that the cleaning stage actually does something, and exactly one natural
categorical column (`zipcode`) to exercise the one-hot encoding path,
without the dozens of categoricals a fuller housing dataset would bring.

## Getting the data

The raw CSV is **not committed to this repository** (see below for why).
Fetch it yourself:

**Option A: browser (simplest)**

1. Sign in at kaggle.com and open
   [House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction).
2. Click **Download**. You get `archive.zip`, which contains `kc_house_data.csv`.
3. Unzip it into this directory:

   ```bash
   unzip ~/Downloads/archive.zip -d data/raw/
   ```

**Option B: Kaggle CLI**

1. From your Kaggle account settings, generate an API token
   ("Create New API Token"). This downloads a `kaggle.json` credentials
   file. Save it to `~/.kaggle/kaggle.json` (do not commit it anywhere).
2. Install the Kaggle CLI, then download and unzip the dataset:

   ```bash
   pip install kaggle
   kaggle datasets download -d harlfoxem/housesalesprediction -p data/raw/ --unzip
   ```

Either way, you should end up with `data/raw/kc_house_data.csv`.

## Why the CSV isn't in git

Kaggle's terms of use generally don't permit redistributing a dataset's raw
files inside your own repository. `data/raw/` is gitignored for this
reason — this README documents how to fetch the file instead of shipping
it.

## Column notes for this pipeline

- `zipcode` looks numeric but is a category, not a magnitude — cast it to
  a string (`df["zipcode"].astype(str)`) before passing it to
  `one_hot_encode` / `house_price_pipeline`'s `cat_labels` argument. Left as
  a raw integer, `np.unique` would sort it in numeric order, and it could
  accidentally be fed to the numeric cleaning functions instead.
- `price` is the regression target (`y`).
- Reasonable numeric feature columns to start with: `sqft_living`,
  `sqft_lot`, `bedrooms`, `bathrooms`, `floors`, `grade`, `yr_built`.
- A natural ratio feature for `assemble_feature_matrix` is
  `sqft_living / sqft_lot` (living area relative to lot size).

## Quick sanity range

On a plain OLS fit (no regularization, no polynomial terms), expect test-set
R² roughly in the 0.6-0.85 range. A number outside that band on your own
run is worth investigating before trusting it — it usually means either a
data leak (statistics fit on more than the train split) or a bug in feature
assembly (e.g. `zipcode` slipping into the numeric cleaning path).
