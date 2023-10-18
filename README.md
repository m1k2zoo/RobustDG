# Generalizability of Adversarial Robustness Under Distribution Shifts
This repository contains the PyTorch implementation of the paper "Generalizability of Adversarial Robustness Under Distribution Shifts" published at the Transactions on Machine Learning Research (TMLR) journal. The paper investigates the interplay between adversarial robustness and domain generalization, and shows that both empirical and certified robustness generalize to unseen domains, even in a real-world medical application.

## Requirements

The code requires the following packages:

- [AutoAttack](https://github.com/fra31/auto-attack) (pip install git+https://github.com/fra31/auto-attack)

## Usage

### Empirical Robustness

To train the models and evaluate the generalization of empirical robustness, run the following command:
```
python -m domainbed.scripts.train_empirical --data_dir ./datasets/ --dataset PACS --algorithm ERM --test_env 0 --steps 300 --output_dir ./logs/
```
This would load the data from `./datasets/PACS/` and does standard ERM training, with environment 0 being the test environment and trains for 300 iterations/steps, the results will be saved in ./logs/ where you will find the best model checkpoint along with clean and robust accuracy (PGD and AutoAttack).

### Certified Robustness

To train the models and evaluate the generalization of certified robustness, run the following command:
```
# code will be added soon.
```

## References

If you use this code or the results in your research, please cite the following paper:

```
@misc{alhamoud2023generalizability,
      title={Generalizability of Adversarial Robustness Under Distribution Shifts}, 
      author={Kumail Alhamoud and Hasan Abed Al Kader Hammoud and Motasem Alfarra and Bernard Ghanem},
      year={2023},
      eprint={2209.15042},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
## Contact:

* **Kumail Alhamoud:**  kumail.hamoud@kaust.edu.sa
* **Hasan Abed Al Kader Hammoud:** hasanabedalkader.hammoud@kaust.edu.sa
