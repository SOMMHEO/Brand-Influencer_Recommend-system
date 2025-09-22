import os, platform
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import mlflow
import mlflow.pytorch
import boto3

load_dotenv(dotenv_path="config/.env")
aws_access_key = os.getenv("aws_accessKey")
aws_secret_key = os.getenv("aws_secretKey")
region_name='ap-northeast-2'
 
s3 = boto3.client('s3',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=region_name) # ec2 사용 시 변경

# Dataset 구성
class NCFDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.users = df['user_uid'].values
        self.items = df['item_uid'].values
        self.labels = df['label'].values

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self,idx):
        return self.users[idx], self.items[idx], self.labels[idx]

##--------------------Models---------------------

# GMF 모델
class GMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super().__init__()
        self.embedding_user = nn.Embedding(num_users, latent_dim)
        self.embedding_item = nn.Embedding(num_items, latent_dim)
        self.output_layer = nn.Linear(latent_dim, 1)

    def forward(self, user, item):
        u = self.embedding_user(user)
        i = self.embedding_item(item)
        out = u * i
        return torch.sigmoid(self.output_layer(out)).squeeze()

# MLP 모델
class MLP(nn.Module):
    def __init__(self, num_users, num_items, layers=[64,32,16,8]):
        super().__init__()
        # embedding_dim = layers[0] // 2
        embedding_dim = int(layers[0] / 2)
        self.embedding_user = nn.Embedding(num_users, embedding_dim)
        self.embedding_item = nn.Embedding(num_items, embedding_dim)

        mlp_modules = []
        input_size = embedding_dim * 2
        for layer_size in layers:
            mlp_modules.append(nn.Linear(input_size, layer_size))
            mlp_modules.append(nn.BatchNorm1d(layer_size)) # 추가
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(0.2)) # 추가
            input_size = layer_size

        self.mlp = nn.Sequential(*mlp_modules)
        self.output_layer = nn.Linear(layers[-1], 1)

    def forward(self, user, item):
        u = self.embedding_user(user)
        i = self.embedding_item(item)
        x = torch.cat([u, i], dim=1)
        out = self.mlp(x)
        return torch.sigmoid(self.output_layer(out)).squeeze()

# GMF + NCF 결합
class NeuMF(nn.Module):
    def __init__(self, gmf_model, mlp_model, mlp_output_dim=8, use_pretrained=False):
        super(NeuMF, self).__init__()
        self.gmf = gmf_model
        self.mlp = mlp_model
        self.use_pretrained = use_pretrained

        gmf_dim = gmf_model.embedding_user.embedding_dim  # latent_dim
        self.fc = nn.Linear(gmf_dim + mlp_output_dim, 1)
        self.sigmoid = nn.Sigmoid()

        if use_pretrained:
            self._init_weights()

    def _init_weights(self):
        # 사전 학습된 가중치를 NeuMF에 복사
        self.gmf.embedding_user.weight.data.copy_(self.gmf.embedding_user.weight)
        self.gmf.embedding_item.weight.data.copy_(self.gmf.embedding_item.weight)
        self.mlp.embedding_user.weight.data.copy_(self.mlp.embedding_user.weight)
        self.mlp.embedding_item.weight.data.copy_(self.mlp.embedding_item.weight)

    def forward(self, user, item):
        gmf_emb_user = self.gmf.embedding_user(user)
        gmf_emb_item = self.gmf.embedding_item(item)
        gmf_output = gmf_emb_user * gmf_emb_item  # [batch_size, latent_dim]

        mlp_emb_user = self.mlp.embedding_user(user)
        mlp_emb_item = self.mlp.embedding_item(item)
        mlp_input = torch.cat([mlp_emb_user, mlp_emb_item], dim=1)
        mlp_output = self.mlp.mlp(mlp_input)  # [batch_size, mlp_output_dim]

        concat = torch.cat([gmf_output, mlp_output], dim=1)  # [batch_size, latent_dim + mlp_output_dim]
        output = self.fc(concat)
        return self.sigmoid(output).squeeze()
    
class EarlyStopping:
    def __init__(self, patience=3, delta=0.0, verbose=True):
        """
        patience: 개선되지 않은 에폭 허용 횟수
        delta: loss가 이 정도 이상 좋아져야 개선으로 간주
        verbose: 로그 출력 여부
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, loss, model):
        if loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.best_model_state = model.state_dict()
            self.counter = 0
            if self.verbose:
                print(f"Train loss improved: {loss:.5f}. saved the model.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"no improvement: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

##---------------model utilities-----------------

def train_gmf_model(train_loader, num_users, num_items, latent_dim, epochs=30, lr=0.001, patience=3, delta=0.0):
    os_name = platform.system()
    if os_name.lower() == 'linux':
        mlflow.set_tracking_uri("file:/home/flexmatch/recommend-system/mlruns")
        mlflow.set_experiment("MLFLOW_Experiment")
                              
    with mlflow.start_run(run_name="GMF_Training"):
        mlflow.log_param('dataset', 'influencer-item interaction dataset')

        # input dim
        mlflow.log_param('num_users', num_users)
        mlflow.log_param('num_items', num_items)

        # General Training Parameters
        mlflow.log_param('patience', patience)
        mlflow.log_param('lr', lr)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param('epochs', epochs)

        # Optimizer and Misc Parameters
        mlflow.log_param("optimizer", "Adam")

        mlflow.log_param("latent_dim", latent_dim)
        # mlflow.log_param("num_parameters", sum(p.numel() for p in model.parameters()))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GMF(num_users, num_items, latent_dim).to(device)

        # gpu 디버깅 코드
        # print(next(model.parameters()).device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True)

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for user, item, label in train_loader:
                user, item, label = user.to(device), item.to(device), label.to(device).float()

                optimizer.zero_grad()
                output = model(user, item).squeeze()
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"[GMF] Epoch {epoch+1}/{epochs} Avg Loss: {avg_loss:.4f}")
            # mlflow.log_metric
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            early_stopping(avg_loss, model)
            if early_stopping.early_stop:
                print("Early stopping (GMF).")
                break

        # EarlyStopping이 저장한 best loss 로깅
        if early_stopping.best_loss is not None:
            mlflow.log_metric("best_loss", early_stopping.best_loss)

        # Best 모델 복원
        if early_stopping.best_model_state:
            model.load_state_dict(early_stopping.best_model_state)

        torch.save(model.state_dict(), "gmf_model.pth")
        mlflow.pytorch.log_model(model, "gmf_model")

        bucket_name = 'flexmatch-data'
        model_path = 'recommendation_system/model/gmf_model.pth'
        s3.upload_file('gmf_model.pth', bucket_name, model_path)
        return model

def train_mlp_model(train_loader, num_users, num_items, layers, epochs=30, lr=0.001, patience=3, delta=0.0):
    with mlflow.start_run(run_name="MLP_Training"):
        mlflow.log_param('dataset', 'influencer-item interaction dataset')

        # input dim
        mlflow.log_param('num_users', num_users)
        mlflow.log_param('num_items', num_items)
        mlflow.log_param('dataset', 'influencer-item interaction dataset')
        
        # General Training Parameters
        mlflow.log_param('patience', patience)
        mlflow.log_param('lr', lr)
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('batch_size', train_loader.batch_size)

        # Model Architecture Parameters
        mlflow.log_param("layers", layers)
        embedding_dim = int(layers[0] / 2) 
        mlflow.log_param("embedding_dim", embedding_dim)
        mlflow.log_param("mlp_layers", layers)
        # mlflow.log_param("num_parameters", sum(p.numel() for p in model.parameters()))

        # Regularization and Activation Parameters
        mlflow.log_param("dropout_rate", 0.2)
        mlflow.log_param("use_batch_norm", True)
        mlflow.log_param("activation_function", "ReLU")

        # Optimizer and Misc Parameters
        mlflow.log_param("optimizer", "Adam")
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # gpu 디버깅 코드
        # print(next(model.parameters()).device)

        model = MLP(num_users, num_items, layers=layers).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True)

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for user, item, label in train_loader:
                user, item, label = user.to(device), item.to(device), label.to(device).float()

                optimizer.zero_grad()
                output = model(user, item).squeeze()
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"[MLP] Epoch {epoch+1}/{epochs} Avg Loss: {avg_loss:.4f}")
            # mlflow.log_metric
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            early_stopping(avg_loss, model)
            if early_stopping.early_stop:
                print("Early stopping (MLP)")
                break
        
        # Best model 복원
        if early_stopping.best_loss is not None:
            mlflow.log_metric("best_loss", early_stopping.best_loss)

        if early_stopping.best_model_state:
            model.load_state_dict(early_stopping.best_model_state)

        torch.save(model.state_dict(), "mlp_model.pth")
        mlflow.pytorch.log_model(model, "mlp_model")

        bucket_name = 'flexmatch-data'
        model_path = 'recommendation_system/model/mlp_model.pth'
        s3.upload_file('mlp_model.pth', bucket_name, model_path)

        return model

def build_neumf_model(num_users, num_items, latent_dim, layers, mlp_output_dim=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gmf_model = GMF(num_users, num_items, latent_dim)
    mlp_model = MLP(num_users, num_items, layers)

    # 사전 학습된 weight 불러오기
    # 만약에 학습은 GPU로 하고 추론은 CPU로 하는 등 모델을 불러와서 사용하는 장치가 바뀔 수 있음. 이때 별도로 사용할 위치를 지정 -> map_location=device 옵션 추가
    gmf_model.load_state_dict(torch.load("gmf_model.pth")) 
    mlp_model.load_state_dict(torch.load("mlp_model.pth"))

    # NeuMF 결합 모델
    neumf = NeuMF(gmf_model, mlp_model, mlp_output_dim=mlp_output_dim, use_pretrained=True).to(device)
    return neumf

def train_neumf_model(train_loader, neumf_model, epochs=20, lr=0.001, patience=3, delta=0.0):
    with mlflow.start_run(run_name="neuMF_Training"):
        mlflow.log_param('Dataset', 'influencer-item interaction dataset')

        mlflow.log_param('patience', patience)
        mlflow.log_param('lr', lr)
        mlflow.log_param('epochs', epochs)
        mlflow.log_param("batch_size", train_loader.batch_size)

        mlflow.log_param("optimizer", "Adam")

        mlflow.log_param('use_pretrained', neumf_model.use_pretrained)
        mlflow.log_param('latent_dim', neumf_model.gmf.embedding_user.embedding_dim)
        mlflow.log_param('mlp_output_dim', neumf_model.fc.in_features - neumf_model.gmf.embedding_user.embedding_dim)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # gpu 디버깅 코드
        # print(next(neumf_model.parameters()).device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(neumf_model.parameters(), lr=lr)
        early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True)

        neumf_model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for user, item, label in train_loader:
                user, item, label = user.to(device), item.to(device), label.to(device).float()

                optimizer.zero_grad()
                output = neumf_model(user, item).squeeze()
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"[NeuMF] Epoch {epoch+1}/{epochs} Avg Loss: {avg_loss:.4f}")
            # mlflow.log_metric
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            # EarlyStopping 확인
            early_stopping(avg_loss, neumf_model)
            if early_stopping.early_stop:
                print("Early stopping (NeuMF)")
                break

        # EarlyStopping이 저장한 best loss 로깅
        if early_stopping.best_loss is not None:
            mlflow.log_metric("best_loss", early_stopping.best_loss)

        # 학습이 끝난 후 best 모델로 복원
        if early_stopping.best_model_state is not None:
            neumf_model.load_state_dict(early_stopping.best_model_state)

        torch.save(neumf_model, "neumf_model.pth") # 모델 자체를 저장
        mlflow.pytorch.log_model(neumf_model, "mlp_model")

        bucket_name = 'flexmatch-data'
        model_path = 'recommendation_system/model/neumf_model.pth'
        s3.upload_file('neumf_model.pth', bucket_name, model_path)
        
        return neumf_model