import sys

from .utils import breaker, prepare_train_and_valid_dataloaders, fit, predict, save_graphs
from .models import get_model


def run():
    
    args_1: tuple = ("--path", "-p")
    args_2: tuple = ("--seed", "-s")
    args_3: tuple = ("--backbone", "-bb")
    args_4: tuple = ("--mode", "-m")
    args_5: tuple = ("--epochs", "-e")
    args_6: tuple = ("--early-stopping", "-es")
    args_7: tuple = ("--batch-size", "-bs")
    args_8: tuple = ("--learning-rate", "-lr")
    args_9: tuple = ("--weight-decay", "-wd")
    args_10: str  = "--size"
    args_11: str  = "--augment" 
    args_12: str  = "--patience-eps"
    args_13: str  = "--test"
    args_14: tuple = ("--kaggle", "-k")

    path: str = "data"
    seed: int = 0
    backbone: str = "mobilenet"
    mode: str = "full"
    epochs: int = 10
    early_stopping: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    wd: float = 0.0
    use_scheduler: bool = False
    patience: int = 5
    eps: float = 1e-8
    size: int = 320
    augment: bool = False
    test: bool = False
    name: str = None
    kaggle: bool = False

    if args_1[0] in sys.argv: path = sys.argv[sys.argv.index(args_1[0]) + 1]
    if args_1[1] in sys.argv: path = sys.argv[sys.argv.index(args_1[1]) + 1]

    if args_2[0] in sys.argv: seed = int(sys.argv[sys.argv.index(args_2[0]) + 1])
    if args_2[1] in sys.argv: seed = int(sys.argv[sys.argv.index(args_2[1]) + 1])

    if args_3[0] in sys.argv: backbone = sys.argv[sys.argv.index(args_3[0]) + 1]
    if args_3[1] in sys.argv: backbone = sys.argv[sys.argv.index(args_3[1]) + 1]

    if args_4[0] in sys.argv: mode = sys.argv[sys.argv.index(args_4[0]) + 1]
    if args_4[1] in sys.argv: mode = sys.argv[sys.argv.index(args_4[1]) + 1]

    if args_5[0] in sys.argv: epochs = int(sys.argv[sys.argv.index(args_5[0]) + 1])
    if args_5[1] in sys.argv: epochs = int(sys.argv[sys.argv.index(args_5[1]) + 1])

    if args_6[0] in sys.argv: early_stopping = int(sys.argv[sys.argv.index(args_6[0]) + 1])
    if args_6[1] in sys.argv: early_stopping = int(sys.argv[sys.argv.index(args_6[1]) + 1])

    if args_7[0] in sys.argv: batch_size = int(sys.argv[sys.argv.index(args_7[0]) + 1])
    if args_7[1] in sys.argv: batch_size = int(sys.argv[sys.argv.index(args_7[1]) + 1])

    if args_8[0] in sys.argv: lr = float(sys.argv[sys.argv.index(args_8[0]) + 1])
    if args_8[1] in sys.argv: lr = float(sys.argv[sys.argv.index(args_8[1]) + 1])

    if args_9[0] in sys.argv: wd = float(sys.argv[sys.argv.index(args_9[0]) + 1])
    if args_9[1] in sys.argv: wd = float(sys.argv[sys.argv.index(args_9[1]) + 1])

    if args_10 in sys.argv: size = int(sys.argv[sys.argv.index(args_10) + 1])

    if args_11 in sys.argv: augment = True

    if args_12 in sys.argv:
        use_scheduler = True
        patience = int(sys.argv[sys.argv.index(args_12) + 1])
        eps = float(sys.argv[sys.argv.index(args_12) + 2])
    
    if args_13 in sys.argv: 
        test = True
        name = sys.argv[sys.argv.index(args_13) + 1]
    
    if args_14[0] in sys.argv or args_14[1] in sys.argv: kaggle = True


    breaker()
    print("Building Model ...")
    model = get_model(seed, mode, backbone)

    if not test:
        breaker()
        print("Preparing Dataloaders ...")

        dataloaders = prepare_train_and_valid_dataloaders(path, mode, batch_size, seed, augment)

        optimizer = model.get_optimizer(lr, wd)
        if use_scheduler:
            scheduler = model.get_plateau_scheduler(optimizer, patience, eps)
        else:
            scheduler = None
        
        L, A, _, _, _ = fit(model=model, optimizer=optimizer, scheduler=scheduler, 
                            epochs=epochs, early_stopping_patience=early_stopping, 
                            dataloaders=dataloaders, verbose=True)
        save_graphs(L, A)

    else:
        assert name is not None, "Enter an Image Name"

        if kaggle:
            label = predict(model, mode, path, size)  #--> Expects Absolute Path to the Image
        else:
            label = predict(model, mode, path + "/test/" + name, size)

        breaker()
        print(f"Label : {label}")
        breaker()

    print("Terminating ...")
    breaker()
