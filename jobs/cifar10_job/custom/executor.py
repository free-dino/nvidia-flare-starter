import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from custom.net import Net

from torchvision import transforms

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable, model_learnable_to_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class CIFAR10Executor(Executor):
    def __init__(
        self,
        epochs: int = 2,
        lr: float = 1e-2,
        momentum: float = 0.9,
        batch_size: int = 4,
        num_workers: int = 1,
        dataset_path: str = "/tmp/nvflare/data/cifar10",
        model_path: str = "/tmp/nvflare/data/cifar10/cifar_net.pth",
        device=DEVICE,
        pre_train_task_name=AppConstants.TASK_GET_WEIGHTS,
        train_task_name=AppConstants.TASK_TRAIN,
        submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
        validate_task_name=AppConstants.TASK_VALIDATION,
        exclude_vars=None
    ):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_path = dataset_path
        self.model_path = model_path
        
        self.pre_train_task_name = pre_train_task_name
        self.train_task_name = train_task_name
        self.submit_model_task_name = submit_model_task_name
        self.validate_task_name = validate_task_name
        self.device = device
        self.exclude_vars = exclude_vars
        
        self.train_dataset = None
        self.valid_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.persistence_manager = None
        self.best_acc = 0.0
        
    def handle_event(self, event_type, fl_ctx):
        if event_type == EventType.START_RUN:
            self.initialize()
            
    def initialize(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])
        
        self.trainset = torchvision.datasets.CIFAR10(
            root = self.dataset_path, 
            train=True, 
            download=True, 
            transform=transform
        )
        
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
        self._n_iterations = len(self.trainloader)
        
        self.testset = torchvision.datasets.CIFAR10(
            root=self.dataset_path,
            train=False,
            download=True,
            transform=transform
        )
        
        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        self.net = Net()
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.SGD(
            self.net.parameters(), 
            lr=self.lr,
            momentum=self.momentum
        )
        
        self._default_train_conf = {
            "train": {
                "model": type(self.net).__name__
            }
        }
        
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.net.state_dict(),
            default_train_conf=self._default_train_conf
        )
        
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        try:
            if task_name == self.pre_train_task_name:
                return self._get_model_weights()
            
            if task_name == self.train_task_name:
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Unable to extract dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)
                
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_error(fl_ctx, f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)
                
                torch_weights = {
                    k: torch.as_tensor(v) for k, v in dxo.data.items()
                }
                self._local_train(fl_ctx, torch_weights)
                
                self._save_local_model(fl_ctx)
                
                return self._get_model_weights(fl_ctx)
            
            if task_name == self.validate_task_name:
                model_owner = "?"
                try:
                    try:
                        dxo = from_shareable(shareable)
                    except:
                        self.log_error(fl_ctx, "Error in extracting dxo from shareable.")
                        return make_reply(ReturnCode.BAD_TASK_DATA)
                    
                    if not dxo.data_kind == DataKind.WEIGHTS:
                        self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.")
                        return make_reply(ReturnCode.BAD_TASK_DATA)
                    
                    model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
                    weights = {
                        k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()
                    }
                    
                    val_accuracy = self._local_validate(fl_ctx, weights)
                    if abort_signal.triggered:
                        return make_reply(ReturnCode.TASK_ABORTED)
                    
                    self.log_info(
                        fl_ctx,
                        f"Accuracy when validating {model_owner}'s model on"
                        f" {fl_ctx.get_identity_name()}"
                        f"s data: {val_accuracy}"
                    )
                    
                    dxo = DXO(data_kind=DataKind.METRICS, data={"val_acc": val_accuracy})
                    return dxo.to_shareable()
                except:
                    self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            elif task_name == self.submit_model_task_name:
                ml = self._load_local_model(fl_ctx)
                
                dxo = model_learnable_to_dxo(ml)
                
                return dxo.to_shareable()
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        
        except Exception as e:
            self.log_exception(fl_ctx, f"Exception is simple trainer: {e}.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            
    def _get_model_weights(self):
        weights = {
            k: v.cpu().numpy() for k, v in self.net.state_dict().items()
        }
        
        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data=weights,
            meta={
                MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations
            }
        )
        return outgoing_dxo.to_shareable()
    
    def _local_train(self, fl_ctx, input_weights):
        self.net.load_state_dict(input_weights)
        self.net.to(self.device)
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                if i % 2000 == 1999:
                    self.log_info(fl_ctx, f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                    running_loss = 0.0
                    
        self.log_info(fl_ctx, "Finished Training")
    
    def _local_validate(self, fl_ctx, input_weights):
        self.net.load_state_dict(input_weights)
        self.net.to(self.device)

        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                
                outputs = self.net(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct // total
        self.log_info(fl_ctx, f"Accuracy of the network on the 10000 test images: {val_accuracy} %")
        return val_accuracy
    
    def _save_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        ml = make_model_learnable(self.net.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), self.model_path)

    def _load_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, "models")
        if not os.path.exists(models_dir):
            return None

        self.persistence_manager = PTModelPersistenceFormatManager(
            data=torch.load(self.model_path), default_train_conf=self._default_train_conf
        )
        ml = self.persistence_manager.to_model_learnable(exclude_vars=self.exclude_vars)
        return ml