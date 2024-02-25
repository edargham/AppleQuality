from config import config
from preprocessing import read_data, plot_specialized
from trainer import run_training_process_ml, run_training_process_nn

from models.machine_learning import build_svc, build_rf, build_adb, build_knn, build_lr, build_nn


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_path = config['data_path']
    data = read_data(data_path)
    data.info()
    print(data.tail())

    # Data visualization
    plot_specialized(data=data, plot_type="histogram",save_path="EDA_visualizations")

    run_training_process_ml(
        build_svc,
        data,
        'Quality',
        'f1'
    )
    
    run_training_process_ml(
        build_rf,
        data,
        'Quality',
        'f1'
    )
    run_training_process_ml(
        build_adb,
        data,
        'Quality',
        'f1'
    )

    run_training_process_ml(
        build_knn,
        data,
        'Quality',
        'f1'
    )

    run_training_process_ml(
        build_lr,
        data,
        'Quality',
        'f1'
    )    

    run_training_process_nn(
        build_nn,
        data,
        'Quality'
    )
