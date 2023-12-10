import glob
import pandas as pd
import dask.dataframe as dd


def files_to_dask(files):
    for file in files:
        df = pd.read_pickle(file)
        df = dd.from_pandas(df, npartitions=1)
        df.to_parquet(file.replace(".pkl", ".parquet"))


def main():
    files = glob.glob("../../data/preprocessed/events-*.pkl")
    files_to_dask(files)


if __name__ == "__main__":
    main()
