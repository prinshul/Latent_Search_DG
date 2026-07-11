import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import PIL
# from torchlars import LARS
import torchvision
from torchvision.models import resnet18
import torchvision.transforms as transforms
print(torch.__version__)
from sklearn.metrics import accuracy_score
import torch.optim as optim
import tqdm
torch.manual_seed(1)
np.random.seed(1)
##################################################### Training f_theta network ###########################################

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

CHECKPOINT_DIR = "./models"

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 350
FEATURE_DIM = 256
IMAGE_SIZE = 224
CLASSES = 7
LR = 1

src_path =['photo', 'art_painting', 'cartoon']
target_path = ['sketch']

# parameters
resnet_train = False
classifier_load = True
train_clf = False
train_vae = False

# data paths 
image_data_path = '../DG/data/pacs_data'
image_label_path = '../DG/data/pacs_label'
image_list_train_photo = '../DG/data/pacs_label/photo_train_kfold.txt'
image_list_train_cartoon = '../DG/data/pacs_label/cartoon_train_kfold.txt'
image_list_train_art ='../DG/data/pacs_label/art_painting_train_kfold.txt'
image_list_valid_photo = '../DG/data/pacs_label/photo_crossval_kfold.txt'
image_list_valid_cartoon = '../DG/data/pacs_label/cartoon_crossval_kfold.txt'
image_list_valid_art = '../DG/data/pacs_label/art_painting_crossval_kfold.txt'
image_list_test_sketch = '../DG/data/pacs_label/sketch_test_kfold.txt'

# Weights 
resnet_weights = '../DG/code/models/resnet/epoch_pacs_resnet_349.pt'
classifier_weights = '../DG/code/models/resnet/resnet_classifier.pt'
classifier_weight_finetuned='../DG/code/vae/single_source/AutoEncoder/models/Vae_128/classifier_finetuned_vae128.pt'
vae_weights = '../DG/code/vae/single_source/AutoEncoder/models/Vae_128/VAEepoch_pacs_resnet_1300.pt'

class PACSDataset(Dataset):
  """ PACS (Photo Art Cartoon Sketch) Dataset """

  def __init__(self, root_dir, image_size, domains=None, transform = None, img_lists = None):
    """
    Arguments:
      img_lists (list of strings): Path to the image list describing the split
      root_dir (string): Path to the root directory of the different domain folders 
      domains (list of strings): List of domains to read the data from. If None,
        all domains are read
      image_size (integer): Image size to resize the images to
    """
    self.root_dir = root_dir
    self.img_lists = img_lists
    if root_dir[-1] != "/":
      self.root_dir = self.root_dir + "/"
    
    self.categories = ['giraffe', 'horse', 'guitar', 'person', 'dog', 'house', 'elephant']

    if domains is None:
      self.domains = ["photo", "sketch", "art_painting", "cartoon"]
    else:
      self.domains = domains
    
    if transform is None:
      self.transform = transforms.ToTensor()
    else:
      self.transform = transform
    # make a list of all the files in the root_dir
    # and read the labels
    self.img_files = []
    self.labels = []
    self.domain_labels = []
    if img_lists is None:
      for domain in self.domains:
        for category in self.categories:
          for image in os.listdir(self.root_dir+domain+'/'+category):
            self.img_files.append(image)
            self.labels.append(self.categories.index(category))
            self.domain_labels.append(self.domains.index(domain))
    else:
      for img_list in img_lists:
        with open(img_list) as img_files:
          for line in img_files:
            img_path, label = line.split()
            domain, category, image = img_path.split("/")
            self.img_files.append(image)
            self.labels.append(self.categories.index(category))
            self.domain_labels.append(self.domains.index(domain))

  def __len__(self):
    return len(self.img_files)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    img_path = self.root_dir + self.domains[self.domain_labels[idx]] + "/" + self.categories[self.labels[idx]] + "/" + self.img_files[idx]
    
    image = PIL.Image.open(img_path)
    label = self.labels[idx]

    return self.transform(image), label


import cv2
import numpy as np

np.random.seed(0)


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample
    
    
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
class cannyedge(object):
    def __init__(self, threshold1 = 100, threshold2 = 200):
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        
    def __call__(self, sample):
        prob = np.random.random_sample()
        if prob < 0.1:
            sample = np.array(sample)
            edges = cv2.Canny(image=sample, threshold1=self.threshold1, threshold2=self.threshold2) # Canny Edge Detection
            edge3c = np.zeros_like(sample)
            edge3c[:,:,0] = edges
            edge3c[:,:,1] = edges
            edge3c[:,:,2] = edges
            invedge = cv2.bitwise_not(edge3c)

            return invedge
        else:
            return sample

color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=IMAGE_SIZE),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.5),
                                              transforms.RandomGrayscale(p=0.5),
                                              transforms.RandomRotation(30),
#                                               GaussianBlur(kernel_size=int(21)), 
                                              cannyedge(),
                                              transforms.ToTensor(),
                                              
#                                               AddGaussianNoise(mean=0, std=0.2),
] )
training_dataset = PACSDataset(image_data_path, IMAGE_SIZE, 
                               ['photo', 'art_painting', 'cartoon'], 
                               transform=data_transforms, 
                               img_lists=[ image_list_train_photo, image_list_train_cartoon, image_list_train_art])

validation_dataset = PACSDataset(image_data_path, IMAGE_SIZE, 
                               ['photo', 'art_painting', 'cartoon'], 
                               transform=data_transforms, 
                               img_lists=[image_list_valid_photo,image_list_valid_cartoon, image_list_valid_art])

training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)


class FNet_PACS_ResNet(nn.Module):
 
  def __init__(self, hidden_layer_neurons, output_latent_dim):
    super(FNet_PACS_ResNet, self).__init__()
    resnet = resnet18(pretrained=True)
    
    self.resnet = nn.Sequential(*list(resnet.children())[:-1])
    self.fc1 = nn.Linear(resnet.fc.in_features,  hidden_layer_neurons)
    self.fc2 = nn.Linear(hidden_layer_neurons, output_latent_dim)
   
  def forward(self, x):
    x = self.resnet(x)
    x = x.squeeze()

    x = self.fc1(x)
    x = F.leaky_relu(x, negative_slope=0.2)

    x = self.fc2(x)
    return x



def train_step(x, labels, model, optimizer, tau):
    model.train()
    optimizer.zero_grad()
    # Forward pass
    z = model(x)

    # Calculate loss
    z = F.normalize(z, dim=1)
    pairwise_labels = torch.flatten(torch.matmul(labels, labels.t()))
    logits = torch.flatten(torch.matmul(z, z.t())) / tau
    loss = F.binary_cross_entropy_with_logits(logits, pairwise_labels)
    pred = torch.sigmoid(logits)   # whether two images are similar or not
    accuracy = (pred.round().float() == pairwise_labels).sum()/float(pred.shape[0])

    # Perform train step
    #optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy()

def validation_step(x, labels, model, tau):
    model.eval()
    with torch.no_grad():
        z = model(x)

        # Calculate loss
        z = F.normalize(z, dim=1)
        pairwise_labels = torch.flatten(torch.matmul(labels, labels.t()))
        logits = torch.flatten(torch.matmul(z, z.t())) / tau
        loss = F.binary_cross_entropy_with_logits(logits, pairwise_labels)
        pred = torch.sigmoid(logits)   # whether two images are similar or not
        accuracy = (pred.round().float() == pairwise_labels).sum()/float(pred.shape[0])    
    return loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy()

def training_loop(model, training_dataloader, validation_dataloader, optimizer, tau=0.1, epochs=200, device=None):
    epoch_wise_loss = []
    epoch_wise_acc = []
    for epoch in range(EPOCHS):
        step_wise_train_loss = []
        step_wise_train_acc = []
        step_wise_val_loss = []
        step_wise_val_acc = []

        for image_batch, labels in training_dataloader:
            image_batch = image_batch.float()
            if dev is not None:
                image_batch, labels = image_batch.to(device), labels.to(device)
            labels_onehot = F.one_hot(labels, CLASSES).float()
            loss, accuracy = train_step(image_batch, labels_onehot, model, optimizer, tau)
            step_wise_train_loss.append(loss)
            step_wise_train_acc.append(accuracy)

        for image_batch, labels in validation_dataloader:
            image_batch = image_batch.float()
            if dev is not None:
                image_batch, labels = image_batch.to(device), labels.to(device)
            labels_onehot = F.one_hot(labels, CLASSES).float()
            loss, accuracy = validation_step(image_batch, labels_onehot, model, tau)
            step_wise_val_loss.append(loss)
            step_wise_val_acc.append(accuracy)

        if (epoch+1)%10 == 0 and epoch>200:
            torch.save({'epoch' : epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'loss': loss}, CHECKPOINT_DIR+"epoch_pacs_resnet_rot30_invcanedge10_"+str(epoch)+".pt")
        epoch_wise_loss.append(np.mean(step_wise_train_loss))
        epoch_wise_acc.append(np.mean(step_wise_train_acc))
        print("epoch: {} training loss: {:.3f} training accuracy: {:.3f} validation loss: {:.3f} validation accuracy: {:.3f}".format(epoch + 1, np.mean(step_wise_train_loss), np.mean(step_wise_train_acc), np.mean(step_wise_val_loss), np.mean(step_wise_val_acc)))

    return epoch_wise_loss, epoch_wise_acc, model

if resnet_train:
    model = FNet_PACS_ResNet(512, FEATURE_DIM)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(dev)
    optimizer = LARS(torch.optim.SGD(model.parameters(), lr=LR))
    epoch_wise_loss, epoch_wise_acc, model = training_loop(model, training_dataloader, validation_dataloader, optimizer, tau=0.1, epochs=EPOCHS, device=dev)
else:

    model = FNet_PACS_ResNet(512, FEATURE_DIM)
    checkpoint = torch.load(resnet_weights)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(dev)

    model.eval()

if classifier_load:
    'Creating model and load weights'
    layers = []
    layers.append(nn.Linear(FEATURE_DIM, FEATURE_DIM))
    layers.append(nn.Linear(FEATURE_DIM, CLASSES))
    classifier = torch.nn.Sequential(*layers).to(dev)
    classifier = classifier.to(dev)
    classifier.load_state_dict(torch.load(classifier_weights))
else:
    'Creating model'
    layers = []
    layers.append(nn.Linear(FEATURE_DIM, FEATURE_DIM))
    layers.append(nn.Linear(FEATURE_DIM, CLASSES))
    classifier = torch.nn.Sequential(*layers).to(dev)
    classifier = classifier.to(dev)

CELoss = nn.CrossEntropyLoss()

if train_clf:
    opt = torch.optim.Adam(classifier.parameters(), lr=0.001)
    for epoch in range(50):
        step_wise_loss = []
        step_wise_accuracy = []

        step_wise_val_loss = []
        step_wise_val_accuracy = []

        classifier.train()
        for image_batch, labels in (training_dataloader):
            image_batch = image_batch.float()
            if dev is not None:
                image_batch, labels = image_batch.to(dev), labels.to(dev)

            # zero the parameter gradients
            opt.zero_grad()

            z = model(image_batch).to(dev)
            pred = classifier(z)
            loss = CELoss(pred, labels)
            accuracy = (pred.argmax(dim=1) == labels).float().sum()/pred.shape[0]
            loss.backward()
            opt.step()

            step_wise_loss.append(loss.detach().cpu().numpy())
            step_wise_accuracy.append(accuracy.detach().cpu().numpy())

        classifier.eval()
        for image_batch, labels in (validation_dataloader):
            image_batch = image_batch.float()
            if dev is not None:
                image_batch, labels = image_batch.to(dev), labels.to(dev)

            z = model(image_batch).to(dev)
            pred = classifier(z)
            loss = CELoss(pred, labels)
            accuracy = (pred.argmax(dim=1) == labels).float().sum()/pred.shape[0]
            loss.backward()
            opt.step()

            step_wise_val_loss.append(loss.detach().cpu().numpy())
            step_wise_val_accuracy.append(accuracy.detach().cpu().numpy())

        print("Epoch " + str(epoch) + " Loss " + str(np.mean(step_wise_loss)) + " Accuracy " + str(np.mean(step_wise_accuracy)))
        print("Val loss " + str(np.mean(step_wise_val_loss)) + " Val accuracy " + str(np.mean(step_wise_val_accuracy)))

#Load single source data again
photo_training_dataset = PACSDataset(image_data_path, IMAGE_SIZE, 
                               ['photo'], 
                               transform=data_transforms, 
                               img_lists=[ image_list_train_photo])

photo_validation_dataset = PACSDataset(image_data_path, IMAGE_SIZE, 
                               ['photo'], 
                               transform=data_transforms, 
                               img_lists=[image_list_valid_photo])

photo_training_dataloader = DataLoader(photo_training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 1)
photo_validation_dataloader = DataLoader(photo_validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 1)


classifier.eval()
preds = []
true = []
with torch.no_grad():
    for image_batch, labels in (photo_training_dataloader):
        image_batch = image_batch.float()
        if dev is not None:
            image_batch, labels = image_batch.to(dev), labels.to(dev)
        z = model(image_batch).to(dev)
        pred = classifier(z)
        preds.append(pred.argmax(dim=1).cpu().numpy())
        true.append(labels.cpu().numpy())
        accuracy = (pred.argmax(dim=1) == labels).float().sum()
        #       step_wise_accuracy.append(accuracy.detach().cpu().numpy())
preds = np.concatenate(preds, axis=0)
true = np.concatenate(true, axis=0)
print(accuracy_score(preds, true))

classifier.eval()
preds = []
true = []
with torch.no_grad():
    for image_batch, labels in (training_dataloader):
        image_batch = image_batch.float()
        if dev is not None:
            image_batch, labels = image_batch.to(dev), labels.to(dev)
        z = model(image_batch).to(dev)
        pred = classifier(z)
        preds.append(pred.argmax(dim=1).cpu().numpy())
        true.append(labels.cpu().numpy())
        accuracy = (pred.argmax(dim=1) == labels).float().sum()
        #       step_wise_accuracy.append(accuracy.detach().cpu().numpy())
preds = np.concatenate(preds, axis=0)
true = np.concatenate(true, axis=0)
print(accuracy_score(preds, true))

test_dataset = PACSDataset( image_data_path, IMAGE_SIZE, 
                           ['sketch'], 
                           img_lists=[image_list_test_sketch])
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 1)


classifier.eval()
preds = []
true = []
with torch.no_grad():
    for image_batch, labels in (test_dataloader):
        image_batch = image_batch.float()
        if dev is not None:
            image_batch, labels = image_batch.to(dev), labels.to(dev)
        z = model(image_batch).to(dev)
        pred = classifier(z)
        preds.append(pred.argmax(dim=1).cpu().numpy())
        true.append(labels.cpu().numpy())
        accuracy = (pred.argmax(dim=1) == labels).float().sum()
        #       step_wise_accuracy.append(accuracy.detach().cpu().numpy())
preds = np.concatenate(preds, axis=0)
true = np.concatenate(true, axis=0)
print(accuracy_score(preds, true))

'Creating model and load weights'
layers = []
layers.append(nn.Linear(FEATURE_DIM, FEATURE_DIM))
layers.append(nn.Linear(FEATURE_DIM, CLASSES))
classifier = torch.nn.Sequential(*layers).to(dev)
classifier = classifier.to(dev)
classifier.load_state_dict(torch.load(classifier_weights))
classifier.eval()

pacs_resnet_fnet = FNet_PACS_ResNet(512, FEATURE_DIM)
checkpoint = torch.load(resnet_weights)
pacs_resnet_fnet.load_state_dict(checkpoint['model_state_dict'])
pacs_resnet_fnet = model.to(dev)

pacs_resnet_fnet.eval()


vae_training_dataset = PACSDataset(image_data_path, IMAGE_SIZE, 
                               ['photo', 'art_painting', 'cartoon'], 
                               transform=data_transforms, 
                               img_lists=[ image_list_train_photo, image_list_train_cartoon, image_list_train_art])

vae_validation_dataset = PACSDataset(image_data_path, IMAGE_SIZE, 
                               ['photo', 'art_painting', 'cartoon'], 
                               transform=data_transforms, 
                               img_lists=[image_list_valid_photo,image_list_valid_cartoon, image_list_valid_art])

vae_training_dataloader = DataLoader(vae_training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 1)
vae_validation_dataloader = DataLoader(vae_validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 1)


class VAE_PACS_ResNet(nn.Module):
    def __init__(self, image_channels=1, h_dim=64, z_dim=32):
        super(VAE_PACS_ResNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),

            # nn.Linear(256, 256),
            # nn.ReLU(),
        )

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)

        self.decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),

            # nn.Linear(256, 256),
            # nn.ReLU(),

            nn.Linear(256, 256)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

# define model and loss
epochs = 2000
vae = VAE_PACS_ResNet().to(dev)
VAEoptim = torch.optim.Adam(vae.parameters(), lr=1e-4)# criterion = nn.MSELoss()
vae

def loss_function(recon_x, x, mu, logvar):
    l2 = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    l1 = F.l1_loss(recon_x, x, reduction='sum')
    return l1 + l2 + KLD, l2, l1, KLD

def trainVAE(epoch):
    vae.train()
    train_loss = 0
    l1_loss = 0
    l2_loss = 0
    kld_loss = 0
    print(epoch)
    for batch_idx, (image_batch, _) in enumerate(vae_training_dataloader):
        image_batch = image_batch.float()
        image_batch = image_batch.to(dev)
        VAEoptim.zero_grad()
        h = pacs_resnet_fnet(image_batch).to(dev)
        h = h.detach()
        recon_batch, mu, logvar = vae(h)
        loss, l2, l1, kld = loss_function(recon_batch, h, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        VAEoptim.step()

        l1_loss += l1.item()
        l2_loss += l2.item()
        kld_loss += kld.item()

    print('====> Epoch: {} Average loss: {:.4f} L1: {:.4f} L2: {:.4f} KLD: {:.4f}'.format(
        epoch, train_loss / len(vae_training_dataloader.dataset), l1_loss / len(vae_training_dataloader.dataset),
        l2_loss / len(vae_training_dataloader.dataset), kld_loss / len(vae_training_dataloader.dataset)))

    loss = train_loss / len(vae_training_dataloader.dataset)
    return loss

if train_vae:
    train_loss = []
    for epoch in range(1, 101):
        loss = trainVAE(epoch)
        train_loss.append(loss)
        if epoch  % 10 == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': vae.state_dict(),
                        'optimizer_state_dict': VAEoptim.state_dict()
                        }, CHECKPOINT_DIR + "VAEepoch_pacs_resnet_" + str(epoch) + ".pt")


    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    # plt.plot(valid_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #     plt.savefig(CHECKPOINT_DIR +'loss.jpg')
    plt.show()
else:
    vae = VAE_PACS_ResNet()
    checkpoint = torch.load(vae_weights)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae = vae.to(dev)

    
#load best weights
# vae = VAE_PACS_ResNet()
# checkpoint = torch.load(vae_weights)
# vae.load_state_dict(checkpoint['model_state_dict'])
# vae = vae.to(dev)
vae.eval()

'Creating model and load weights'
layers = []
layers.append(nn.Linear(FEATURE_DIM, FEATURE_DIM))
layers.append(nn.Linear(FEATURE_DIM, CLASSES))
classifier = torch.nn.Sequential(*layers).to(dev)
classifier = classifier.to(dev)
classifier.load_state_dict(torch.load(classifier_weights))
classifier.eval()

model = FNet_PACS_ResNet(512, FEATURE_DIM)
checkpoint = torch.load(resnet_weights)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(dev)

model.eval()

#Load single source data again
photo_training_dataset = PACSDataset(image_data_path, IMAGE_SIZE, 
                               ['photo'], 
                               transform=data_transforms, 
                               img_lists=[ image_list_train_photo])

photo_validation_dataset = PACSDataset(image_data_path, IMAGE_SIZE, 
                               ['photo'], 
                               transform=data_transforms, 
                               img_lists=[image_list_valid_photo])

photo_training_dataloader = DataLoader(photo_training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 1)
photo_validation_dataloader = DataLoader(photo_validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 1)


classifier.eval()
model.eval()
preds = []
true = []
with torch.no_grad():
    for image_batch, labels in (photo_training_dataloader):
        image_batch = image_batch.float()
        if dev is not None:
            image_batch, labels = image_batch.to(dev), labels.to(dev)
        z = model(image_batch).to(dev)
        pred = classifier(z)
        preds.append(pred.argmax(dim=1).cpu().numpy())
        true.append(labels.cpu().numpy())
        accuracy = (pred.argmax(dim=1) == labels).float().sum()
        #       step_wise_accuracy.append(accuracy.detach().cpu().numpy())
preds = np.concatenate(preds, axis=0)
true = np.concatenate(true, axis=0)
print("classifier_photo_novae",accuracy_score(preds, true))

#load best weights
# vae = VAE_PACS_ResNet()
# checkpoint = torch.load(vae_weights)
# vae.load_state_dict(checkpoint['model_state_dict'])
# vae = ae.to(dev)
# vae.eval()

'Creating model and load weights'
layers = []
layers.append(nn.Linear(FEATURE_DIM, FEATURE_DIM))
layers.append(nn.Linear(FEATURE_DIM, CLASSES))
classifier = torch.nn.Sequential(*layers).to(dev)
classifier = classifier.to(dev)
classifier.load_state_dict(torch.load(classifier_weights))
classifier.eval()

model = FNet_PACS_ResNet(512, FEATURE_DIM)
checkpoint = torch.load(resnet_weights)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(dev)

model.eval()


#Load single source data again
photo_training_dataset = PACSDataset(image_data_path, IMAGE_SIZE, 
                               ['photo'], 
                               transform=data_transforms, 
                               img_lists=[ image_list_train_photo])

photo_validation_dataset = PACSDataset(image_data_path, IMAGE_SIZE, 
                               ['photo'], 
                               transform=data_transforms, 
                               img_lists=[image_list_valid_photo])

photo_training_dataloader = DataLoader(photo_training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 1)
photo_validation_dataloader = DataLoader(photo_validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 1)
# test on source data# test target dataset
classifier.eval()
model.eval()
vae.eval()
preds = []
true = []
with torch.no_grad():
    for image_batch, labels in (photo_training_dataloader):
        image_batch = image_batch.float()
        if dev is not None:
            image_batch, labels = image_batch.to(dev), labels.to(dev)
        z = model(image_batch).to(dev)
        z,_, _ = vae(z)
        pred = classifier(z)
        preds.append(pred.argmax(dim=1).cpu().numpy())
        true.append(labels.cpu().numpy())
        accuracy = (pred.argmax(dim=1) == labels).float().sum()
        #       step_wise_accuracy.append(accuracy.detach().cpu().numpy())
preds = np.concatenate(preds, axis=0)
true = np.concatenate(true, axis=0)
print("classifiier_photo_vae",accuracy_score(preds, true))

# test target dataset
classifier.eval()
model.eval()
vae.eval()
preds = []
true = []
with torch.no_grad():
    for image_batch, labels in (test_dataloader):
        image_batch = image_batch.float()
        if dev is not None:
            image_batch, labels = image_batch.to(dev), labels.to(dev)
        z = model(image_batch).to(dev)
        z,_,_ = vae(z)
        pred = classifier(z)
        preds.append(pred.argmax(dim=1).cpu().numpy())
        true.append(labels.cpu().numpy())
        accuracy = (pred.argmax(dim=1) == labels).float().sum()
        #       step_wise_accuracy.append(accuracy.detach().cpu().numpy())
preds = np.concatenate(preds, axis=0)
true = np.concatenate(true, axis=0)
print("classifier_target_vae",accuracy_score(preds, true))

'Creating model and load weights'
layers = []
layers.append(nn.Linear(FEATURE_DIM, FEATURE_DIM))
layers.append(nn.Linear(FEATURE_DIM, CLASSES))
classifier = torch.nn.Sequential(*layers).to(dev)
classifier = classifier.to(dev)
classifier.load_state_dict(torch.load(classifier_weights))
classifier.eval()

model = FNet_PACS_ResNet(512, FEATURE_DIM)
checkpoint = torch.load(resnet_weights)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(dev)

model.eval()

vae = VAE_PACS_ResNet()
checkpoint = torch.load(vae_weights)
vae.load_state_dict(checkpoint['model_state_dict'])
vae = vae.to(dev)
vae.eval()


layers = []
layers.append(nn.Linear(FEATURE_DIM, FEATURE_DIM))
layers.append(nn.Linear(FEATURE_DIM, CLASSES))
classifier_finetuned = torch.nn.Sequential(*layers).to(dev)
classifier_finetuned = classifier_finetuned.to(dev)
classifier_finetuned.load_state_dict(torch.load(classifier_weights))
classifier_finetuned.eval()

# retrain model
# classifier_finetuned = torch.load(classifier_weights)
opt = torch.optim.Adam(classifier_finetuned.parameters(), lr=0.01)
classifier_finetuned.train()
CELoss = nn.CrossEntropyLoss()
for epoch in range(50):
    step_wise_loss = []
    step_wise_accuracy = []

    step_wise_val_loss = []
    step_wise_val_accuracy = []

    classifier_finetuned.train()
    for image_batch, labels in (vae_training_dataloader):
        image_batch = image_batch.float()
        if dev is not None:
            image_batch, labels = image_batch.to(dev), labels.to(dev)

        # zero the parameter gradients
        opt.zero_grad()

        z = pacs_resnet_fnet(image_batch).to(dev)
        z, _, _ = vae(z)
        pred = classifier_finetuned(z.reshape(-1, 256))
        loss = CELoss(pred, labels)
        accuracy = (pred.argmax(dim=1) == labels).float().sum()/pred.shape[0]
        loss.backward()
        opt.step()

        step_wise_loss.append(loss.detach().cpu().numpy())
        step_wise_accuracy.append(accuracy.detach().cpu().numpy())

    classifier_finetuned.eval()
    for image_batch, labels in (vae_validation_dataloader):
        image_batch = image_batch.float()
        if dev is not None:
            image_batch, labels = image_batch.to(dev), labels.to(dev)

        z = model(image_batch).to(dev)
        pred = classifier_finetuned(z)
        loss = CELoss(pred, labels)
        accuracy = (pred.argmax(dim=1) == labels).float().sum()/pred.shape[0]
        loss.backward()
        opt.step()

        step_wise_val_loss.append(loss.detach().cpu().numpy())
        step_wise_val_accuracy.append(accuracy.detach().cpu().numpy())
  
#      print("Epoch " + str(epoch) + " Loss " + str(np.mean(step_wise_loss)) + " Accuracy " + str(np.mean(step_wise_accuracy)))
#      print("Val loss " + str(np.mean(step_wise_val_loss)) + " Val accuracy " + str(np.mean(step_wise_val_accuracy)))
print("finetuned_classifier",np.mean(step_wise_accuracy))


classifier_finetuned.eval()
model.eval()
vae.eval()
preds = []
true = []
with torch.no_grad():
    for image_batch, labels in (photo_training_dataloader):
        image_batch = image_batch.float()
        if dev is not None:
            image_batch, labels = image_batch.to(dev), labels.to(dev)
        z = model(image_batch).to(dev)
        z,_, _ = vae(z)
        pred = classifier_finetuned(z)
        preds.append(pred.argmax(dim=1).cpu().numpy())
        true.append(labels.cpu().numpy())
        accuracy = (pred.argmax(dim=1) == labels).float().sum()
        #       step_wise_accuracy.append(accuracy.detach().cpu().numpy())
preds = np.concatenate(preds, axis=0)
true = np.concatenate(true, axis=0)
print("finetuned_classifier_photo_vae",accuracy_score(preds, true))

# test target of classifier finetuned
# test target dataset
classifier_finetuned.eval()
model.eval()
vae.eval()
preds = []
true = []
with torch.no_grad():
    for image_batch, labels in (test_dataloader):
        image_batch = image_batch.float()
        if dev is not None:
            image_batch, labels = image_batch.to(dev), labels.to(dev)
        z = model(image_batch).to(dev)
        z,_,_ = vae(z)
        pred = classifier_finetuned(z)
        preds.append(pred.argmax(dim=1).cpu().numpy())
        true.append(labels.cpu().numpy())
        accuracy = (pred.argmax(dim=1) == labels).float().sum()
        #       step_wise_accuracy.append(accuracy.detach().cpu().numpy())
preds = np.concatenate(preds, axis=0)
true = np.concatenate(true, axis=0)
print("finetuned_clf_target_vae",accuracy_score(preds, true))

# checking target after passing through vae using classifier_finetuned
'Creating model and load weights'
# layers = []
# layers.append(nn.Linear(FEATURE_DIM, FEATURE_DIM))
# layers.append(nn.Linear(FEATURE_DIM, CLASSES))
# classifier_finetuned = torch.nn.Sequential(*layers).to(dev)
# classifier_finetuned = classifier_finetuned.to(dev)
# classifier_finetuned.load_state_dict(torch.load(classifier_weight_finetuned))
# classifier_finetuned.eval()

# test target of classifier finetuned
# test target dataset
classifier_finetuned.eval()
model.eval()
vae.eval()
preds = []
true = []
with torch.no_grad():
    for image_batch, labels in (photo_training_dataloader):
        image_batch = image_batch.float()
        if dev is not None:
            image_batch, labels = image_batch.to(dev), labels.to(dev)
        z = model(image_batch).to(dev)
#         z,_,_ = vae(z)
        pred = classifier_finetuned(z)
        preds.append(pred.argmax(dim=1).cpu().numpy())
        true.append(labels.cpu().numpy())
        accuracy = (pred.argmax(dim=1) == labels).float().sum()
        #       step_wise_accuracy.append(accuracy.detach().cpu().numpy())
preds = np.concatenate(preds, axis=0)
true = np.concatenate(true, axis=0)
print("Finetuned_class_photo_no_vae",accuracy_score(preds, true))

classifier_finetuned.eval()
model.eval()
vae.eval()
preds = []
true = []
with torch.no_grad():
    for image_batch, labels in (training_dataloader):
        image_batch = image_batch.float()
        if dev is not None:
            image_batch, labels = image_batch.to(dev), labels.to(dev)
        z = model(image_batch).to(dev)
#         z,_,_ = vae(z)
        pred = classifier_finetuned(z)
        preds.append(pred.argmax(dim=1).cpu().numpy())
        true.append(labels.cpu().numpy())
        accuracy = (pred.argmax(dim=1) == labels).float().sum()
        #       step_wise_accuracy.append(accuracy.detach().cpu().numpy())
preds = np.concatenate(preds, axis=0)
true = np.concatenate(true, axis=0)
print("Finetuned_clf_all_no_vae",accuracy_score(preds, true))

# checking target after passing through vae using classifier_finetuned 
classifier_finetuned.eval()
model.eval()
vae.eval()
preds = []
true = []
with torch.no_grad():
    for image_batch, labels in (test_dataloader):
        image_batch = image_batch.float()
        if dev is not None:
            image_batch, labels = image_batch.to(dev), labels.to(dev)
        z = model(image_batch).to(dev)
#         z,_,_ = vae(z)
        pred = classifier_finetuned(z)
        preds.append(pred.argmax(dim=1).cpu().numpy())
        true.append(labels.cpu().numpy())
        accuracy = (pred.argmax(dim=1) == labels).float().sum()
        #       step_wise_accuracy.append(accuracy.detach().cpu().numpy())
preds = np.concatenate(preds, axis=0)
true = np.concatenate(true, axis=0)
print("finetuned_clf_target_no_vae",accuracy_score(preds, true))



beta = 0.001
M = 10000
classifier_finetuned.eval()
runs = 1
# elbow = ptr
accuracy_per_run = []
Llist = []
Zlist = []
predList = []
predProbs = []
true = []
k = 0
W = 5
for run in range(1):
    print('run:', run)
    step_wise_accuracy = []

    for image_batch, labels in test_dataloader:
        image_batch = image_batch.float()
        if dev is not None:
            image_batch, labels = image_batch.to(dev), labels.to(dev)

        h = model(image_batch).to(dev)
        h = h.detach()
        batches = 32  # int(len(image_batch)/1)

        for batch in (range(batches)):
            lbl = labels[batch * 1:(batch + 1) * 1]
            x_real = h[batch * 1:(batch + 1) * 1]
            no_1hot = lbl
            true.append(no_1hot.detach().cpu().numpy()[0])
            lbl = F.one_hot(lbl, CLASSES).float()

            zparam, _, _ = vae.encode(x_real)
#             zparam = torch.randn(1, 256).to(dev)
            zparam = zparam.detach().requires_grad_(True)
            # zoptim = LARS(torch.optim.SGD([zparam], lr=beta,momentum=0.9, nesterov=True))
            zoptim = torch.optim.Adam([zparam], lr=beta)

            Zparam = []
            L_s = []
            preds = []
            predP = []


            for itr in range(0, M):  # projection
                zoptim.zero_grad()
                xhat = vae.decode(zparam).to(dev)

                Zparam.append(xhat.detach().cpu().numpy())
                preds.append(classifier_finetuned(xhat.reshape(1, -1)).argmax(dim=1).cpu().numpy()[0])
                predP.append(classifier_finetuned(xhat.reshape(1, -1)).detach().cpu().numpy())
                xhat = xhat.view(1, FEATURE_DIM)
                x_real = x_real.view(1, FEATURE_DIM)

                xhat = F.normalize(xhat, dim=1)
                x_real = F.normalize(x_real, dim=1)
                xhat = xhat.view(FEATURE_DIM)
                x_real = x_real.view(FEATURE_DIM)
                fnetloss = 1 - torch.dot(xhat, x_real)
                fnetloss.backward()
                zoptim.step()
                l = fnetloss.detach().cpu().numpy()
                u_param = zparam.detach().cpu().numpy()
                L_s.append(l)
            k += 1
            L_s = np.asarray(L_s)
            Llist.append(L_s)
            Zparam = np.asarray(Zparam)
            Zlist.append(Zparam)
            preds = np.asarray(preds)
            predList.append(preds)
            predP = np.asarray(predP)
            predProbs.append(predP)
            smooth_L_s = np.cumsum(np.insert(L_s, 0, 0))
            s_vec = (smooth_L_s[W:] - smooth_L_s[:-W]) / W

            double_derivative = []
            s_len = len(s_vec)
            for i in range(1, s_len - 1):
                double_derivative.append(s_vec[i + 1] + s_vec[i - 1] - 2 * s_vec[i])
            double_derivative = np.asarray(double_derivative)

            if L_s[-1] > 0.5:
                z_in = x_real
            else:
                zstar = torch.from_numpy(Zparam[np.argmax(double_derivative)])
                z_in = zstar  # ae.decode(zstar.to(dev))

            z_in = z_in.view(-1, FEATURE_DIM)
            pred = classifier_finetuned(z_in.to(dev))
            accuracy = (pred.argmax(dim=1) == no_1hot).float().sum() / pred.shape[0]
            step_wise_accuracy.append(accuracy.detach().cpu().numpy())

    print("Latent search",np.mean(step_wise_accuracy))
    accuracy_per_run.append(np.mean(step_wise_accuracy))
print(true)
for i in range(32):
    figure, axis = plt.subplots(2, 1)
    axis[0].plot(Llist[i])
    axis[1].plot(np.array([true[i]] * 5000))
    axis[1].plot(predList[i])

    plt.savefig(CHECKPOINT_DIR +'loss_latentsearch' + str(i) + '.png')
    plt.close('all')
