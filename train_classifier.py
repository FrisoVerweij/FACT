import argparse
import time
import yaml
from utils import *


def train(config, seed):
    '''
    Given a set of config parameters, train a neural network
    :config: dicionary hyperparameters specified in the config file
    :param seed: seed for reproducibility
    :return: nn.module: model
    '''
    # Set seeds to make test reproducible
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(config['device'])

    # Select the dataloaders for the right dataset
    traindata_loader, testdata_loader = select_dataloader(config)

    # Select model, optimizer, scheduler and loss function
    model = select_classifier(config).to(device)
    #print(model)

    optimizer = select_optimizer(config, model)
    loss_module = nn.CrossEntropyLoss()

    # Record current time to record training time
    start_time = time.time()
    train_accuracy = 0
    for epoch in range(config['epochs']):
        batch_count = 0
        model.train()
        for inputs, targets in traindata_loader:
            # Prepare data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Calculate model output and loss
            predictions, prediction_probabilities = model(inputs)

            loss = loss_module(predictions, targets)

            accuracyPredictions = torch.argmax(prediction_probabilities, dim=-1)
            correct = (accuracyPredictions == targets).sum().item()
            train_accuracy += correct / len(targets)

            # Execute backwards propagation
            model.zero_grad()
            loss.backward()
            optimizer.step()

            if not config['no_print']:
                print("[Train Epoch %d/%d] [Batch %d] time: %4.4f [loss: %f]" % (
                    epoch + 1, config['epochs'], batch_count, time.time() - start_time,
                    loss.item()))

            batch_count += 1

        # Calculate test loss
        model.eval()
        with torch.no_grad():
            total_comparisons = 0
            total_correct = 0
            for inputs, targets in testdata_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                _, prediction_probabilities = model(inputs)

                predictions = torch.argmax(prediction_probabilities, dim=-1)
                total_comparisons += len(targets)
                total_correct += (predictions == targets).sum().item()

            accuracy = total_correct / total_comparisons
            print("[Test Epoch %d/%d] [test accuracy: %f, train accuracy: %f]" % (epoch + 1, config['epochs'], accuracy, train_accuracy/batch_count))

        train_accuracy = 0

    torch.save(model.state_dict(), config['save_dir'] + config['model_name'])
    return model



if __name__ == "__main__":
    # Create parser to get hyperparameters from user
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/mnist_3_8.yml')
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"))
    config = to_classifier_config(config)
    print(config)
    train(config, 1)