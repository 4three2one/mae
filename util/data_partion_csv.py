

def partion_csv(file):
    import pandas as pd
    df=pd.read_csv(file)
    train=df.sample(frac=0.9)
    val = df[~df.index.isin(train.index)]
    train.to_csv("train.csv")
    val.to_csv("test.csv")
    pass


if __name__ == '__main__':
    file=f"D:\windows_down\deepweeds_dataset\labels.csv"
    partion_csv(file)