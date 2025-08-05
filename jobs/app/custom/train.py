import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from net import Net

import nvflare.client as flare
from nvflare.app_common.app_constant import ModelName

CIFAR10_ROOT = "/tmp/nvflare/data/cifar10"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=CIFAR10_ROOT, nargs="?")
    parser.add_argument("--batch_size", type=int, default=4, nargs="?")
    parser.add_argument("--num_workers", type=int, default=1, nargs="?")
    parser.add_argument("--local_epochs", type=int, default=2, nargs="?")
    parser.add_argument("--model_path", type=str, default=f"{CIFAR10_ROOT}/cifar_net.pth", nargs="?")
    return parser.parse_args()


def main():
    args = define_parser()

    dataset_path = args.dataset_path
    batch_size = args.batch_size
    num_workers = args.num_workers
    local_epochs = args.local_epochs
    model_path = args.model_path

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), 
            (0.5, 0.5, 0.5)
        )
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=dataset_path, 
        train=True, 
        download=True, 
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    testset = torchvision.datasets.CIFAR10(
        root=dataset_path, 
        train=False, download=True, 
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    net = Net()
    best_accuracy = 0.0

    def evaluate(input_weights):
        net = Net()
        net.load_state_dict(input_weights)
        net.to(DEVICE)

        correct = 0
        total = 0

        with torch.no_grad():
            for data in testloader:

                images, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                outputs = net(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct // total

    flare.init()

    while flare.is_running():

        input_model = flare.receive()
        client_id = flare.get_site_name()

        if flare.is_train():
            print(f"({client_id}) current_round={input_model.current_round}, total_rounds={input_model.total_rounds}")

            net.load_state_dict(input_model.params)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

            net.to(DEVICE)
            steps = local_epochs * len(trainloader)
            for epoch in range(local_epochs):

                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):

                    inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                    optimizer.zero_grad()

                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    if i % 2000 == 1999:
                        print(f"({client_id}) [{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                        running_loss = 0.0

            print(f"({client_id}) Finished Training")

            local_accuracy = evaluate(net.state_dict())
            print(f"({client_id}) Evaluating local trained model. Accuracy on the 10000 test images: {local_accuracy}")
            if local_accuracy > best_accuracy:
                best_accuracy = local_accuracy
                torch.save(net.state_dict(), model_path)

            accuracy = evaluate(input_model.params)
            print(
                f"({client_id}) Evaluating received model for model selection. Accuracy on the 10000 test images: {accuracy}"
            )

            output_model = flare.FLModel(
                params=net.cpu().state_dict(),
                metrics={"accuracy": accuracy},
                meta={"NUM_STEPS_CURRENT_ROUND": steps},
            )

            flare.send(output_model)

        elif flare.is_evaluate():
            accuracy = evaluate(input_model.params)
            flare.send(flare.FLModel(metrics={"accuracy": accuracy}))

        elif flare.is_submit_model():
            model_name = input_model.meta["submit_model_name"]
            if model_name == ModelName.BEST_MODEL:
                try:
                    weights = torch.load(model_path)
                    net = Net()
                    net.load_state_dict(weights)
                    flare.send(flare.FLModel(params=net.cpu().state_dict()))
                except Exception as e:
                    raise ValueError("Unable to load best model") from e
            else:
                raise ValueError(f"Unknown model_type: {model_name}")


if __name__ == "__main__":
    main()
