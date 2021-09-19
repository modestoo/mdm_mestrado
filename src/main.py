from loguru import logger

from data_process.data_init import DataInit
from data_process.data_preprocessing import preprocessing_data
from data_process.model_train import train_lr_model, train_xgboost_model

# Iniciando o pre-processamento
logger.info("[ + ] Realizando o pre-processamento dos dados.")
preprocessing_data()

# Iniciando o(s) dado(s)
data_init = DataInit()
models_to_run = [
    "model_1",
    "model_2",
    "model_3",
    "model_4",
    "model_5",
]

# Hiperparâmetros LR
alpha = 0.00001
penalty = "ElasticNet"

# Hiperparâmetros XGBoost
n_estimators = 100
max_depth = 6

for model in models_to_run:
    (
        model_train_stack,
        model_test_stack,
        train_labels,
        test_labels,
    ) = data_init.model_start(model)

    # Executando o(s) modelo(s)
    logger.info(f"[ + ] Executando a inferência do LR no modelo {model}.")
    lr_log_loss, lr_accuracy_score, lr_f1_score = train_lr_model(
        model,
        alpha,
        penalty,
        model_train_stack,
        train_labels,
        model_test_stack,
        test_labels,
    )

    logger.info(f"[ + ] Executando a inferência do XGBoost no modelo {model}.")
    xgboost_log_loss, xgboost_accuracy_score, xgboost_f1_score = train_xgboost_model(
        model,
        n_estimators,
        max_depth,
        model_train_stack,
        train_labels,
        model_test_stack,
        test_labels,
    )

    # Salva os resultados em um arquivo
    logger.info(
        f"[ + ] Escrevendo o resultado do modelo {model} no arquivo de resultados."
    )
    with open("models/result_models.txt") as file:
        file.write(
            f"Model: {model}; \n \
                lr_log_loss: {lr_log_loss}, \
                lr_accuracy_score: {lr_accuracy_score}, \
                lr_f1_score: {lr_f1_score}, \n \
                xgboost_log_loss: {xgboost_log_loss}, \
                xgboost_accuracy_score: {xgboost_accuracy_score}, \
                xgboost_f1_score: {xgboost_f1_score} \n"
        )
