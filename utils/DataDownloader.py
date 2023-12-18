import os

import wget

url_dict = {
    "train_pbmc": "https://www.dropbox.com/s/wk5zewf2g1oat69/train_pbmc.h5ad?dl=1",
    "valid_pbmc": "https://www.dropbox.com/s/nqi971n0tk4nbfj/valid_pbmc.h5ad?dl=1",

    "train_hpoly": "https://www.dropbox.com/s/7ngt0hv21hl2exn/train_hpoly.h5ad?dl=1",
    "valid_hpoly": "https://www.dropbox.com/s/bp6geyvoz77hpnz/valid_hpoly.h5ad?dl=1",

    "train_salmonella": "https://www.dropbox.com/s/9ozdwdi37wrz9r1/train_salmonella.h5ad?dl=1",
    "valid_salmonella": "https://www.dropbox.com/s/z5jnq4nthierdgq/valid_salmonella.h5ad?dl=1",

    "train_species": "https://www.dropbox.com/s/eprgwhd98c9quiq/train_species.h5ad?dl=1",
    "valid_species": "https://www.dropbox.com/s/bwq18z0mzy6h5d7/valid_species.h5ad?dl=1",

    "train_study": "https://www.dropbox.com/s/75cdaxyn9ldpeqz/train_study.h5ad?dl=1",
    "valid_study": "https://www.dropbox.com/s/klyy7ybeiwo0u8a/valid_study.h5ad?dl=1",

    "train_zheng": "https://www.dropbox.com/s/deajaxpmrxnjatj/train_zheng.h5ad?dl=1",

    "pancreas": "https://www.dropbox.com/s/qj1jlm9w10wmt0u/pancreas.h5ad?dl=1",
    "bbknn": "https://www.dropbox.com/s/3kprctbxyxnlrgt/bbknn.h5ad?dl=1",
    "cca": "https://www.dropbox.com/s/mxk9mbhelt7kn22/cca.h5ad?dl=1",
    "mnn": "https://www.dropbox.com/s/n4vl10h7zw7m6tl/mnn.h5ad?dl=1",
    "scanorama": "https://www.dropbox.com/s/j8fti1txfa57dvm/scanorama.h5ad?dl=1",

    "MouseAtlas.subset": "https://www.dropbox.com/s/zkss8ds1pi0384p/MouseAtlas.subset.h5ad?dl=1"

}


def download_data(data_name, key=None):
    data_path = "data/"
    if key is None:
        train_path = os.path.join(data_path, f"train_{data_name}.h5ad")
        valid_path = os.path.join(data_path, f"valid_{data_name}.h5ad")

        train_url = url_dict[f"train_{data_name}"]
        valid_url = url_dict[f"valid_{data_name}"]

        if not os.path.exists(train_path):
            wget.download(train_url, train_path)
        if not os.path.exists(valid_path):
            wget.download(valid_url, valid_path)
    else:
        data_path = os.path.join(data_path, f"{key}.h5ad")
        data_url = url_dict[key]

        if not os.path.exists(data_path):
            wget.download(data_url, data_path)
    print(f"{data_name} data has been downloaded and saved in {data_path}")


def main():
    data_names = ["pbmc", "hpoly", "salmonella", "species", "study"]
    for data_name in data_names:
        download_data(data_name)
    keys = ["train_zheng", "pancreas", "bbknn", "cca", "mnn", "scanorama", "MouseAtlas.subset"]
    for key in keys:
        download_data(None, key)


if __name__ == '__main__':
    main()
