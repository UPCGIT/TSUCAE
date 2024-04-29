import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torchvision.transforms as transforms
from TSUCAE_model import Unmixing
from utility import load_HSI, hyperVca, load_data, reconstruction_SADloss
from utility import plotAbundancesGT, plotAbundancesSimple, plotEndmembersAndGT, reconstruct
import time
import os
import pandas as pd
import random
from scipy.spatial.distance import cosine
from skimage.filters import threshold_otsu
from sklearn.metrics.pairwise import cosine_similarity

start_time = time.time()

seed = 1
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def cos_dist(x1, x2):
    return cosine(x1, x2)


def binning_mapping(similarity_matrix, threshold, ranges):
    below_threshold_mask = similarity_matrix < threshold
    above_threshold_mask = similarity_matrix >= threshold

    percentiles_below = np.percentile(similarity_matrix[below_threshold_mask], [20, 40, 60, 80])
    percentiles_above = np.percentile(similarity_matrix[above_threshold_mask], [20, 40, 60, 80])

    mapped_array = np.empty_like(similarity_matrix)

    for i in range(5):
        lower = percentiles_below[i - 1] if i > 0 else similarity_matrix[below_threshold_mask].min()
        upper = percentiles_below[i] if i < 4 else similarity_matrix[below_threshold_mask].max()
        mask = (similarity_matrix >= lower) & (similarity_matrix <= upper) & below_threshold_mask
        mapped_array[mask] = ranges[i][0] + (similarity_matrix[mask] - lower) / (upper - lower) * (
                ranges[i][1] - ranges[i][0])

    for i in range(5):
        lower = percentiles_above[i - 1] if i > 0 else similarity_matrix[above_threshold_mask].min()
        upper = percentiles_above[i] if i < 4 else similarity_matrix[above_threshold_mask].max()
        mask = (similarity_matrix >= lower) & (similarity_matrix <= upper) & above_threshold_mask
        mapped_array[mask] = ranges[i + 5][0] + (similarity_matrix[mask] - lower) / (upper - lower) * (
                ranges[i + 5][1] - ranges[i + 5][0])

    return mapped_array


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

datasetnames = {'Urban': 'Urban4',
                'Samson': 'Samson',
                'dc': 'DC',
                }
dataset = "Samson"  # Replace the data set here
hsi = load_HSI("Datasets/" + datasetnames[dataset] + ".mat")
data = hsi.array()
endmember_number = hsi.gt.shape[0]
col = hsi.cols
band_number = data.shape[1]

batch_size = 1
EPOCH = 600
num_runs = 1

if dataset == "Samson":
    drop_out = 0.25
    learning_rate = 0.03
    step_size = 25
    gamma = 0.7
    weight_decay = 1e-3
if dataset == "Urban":
    drop_out = 0.25
    learning_rate = 0.01
    step_size = 40
    gamma = 0.7
    weight_decay = 5e-4
if dataset == "dc":
    drop_out = 0.25
    learning_rate = 0.005
    step_size = 50
    gamma = 0.5
    weight_decay = 1e-3

end = []
end2 = []
abu = []
r = []

output_path = 'Results'
method_name = 'TSUCAE'
mat_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'mat'
endmember_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'endmember'
abundance_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'abundance'
if not os.path.exists(mat_folder):
    os.makedirs(mat_folder)
if not os.path.exists(endmember_folder):
    os.makedirs(endmember_folder)
if not os.path.exists(abundance_folder):
    os.makedirs(abundance_folder)

for run in range(1, num_runs + 1):
    VCA_endmember, _, _ = hyperVca(data.T, endmember_number)
    VCA_endmember = torch.from_numpy(VCA_endmember)
    GT_endmember = hsi.gt.T
    endmember_init = VCA_endmember.unsqueeze(2).unsqueeze(3).float()
    print('Start training!', 'run:', run)

    abundance_GT = torch.from_numpy(hsi.abundance_gt)
    abundance_GT = torch.reshape(abundance_GT, (col * col, endmember_number))
    original_HSI = torch.from_numpy(data)
    original_HSI = torch.reshape(original_HSI.T, (band_number, col, col))
    abundance_GT = torch.reshape(abundance_GT.T, (endmember_number, col, col))

    image = np.array(original_HSI)

    """
    # Calculate the cosine similarity to get similarity matrix
    # The closer the cosine distance is to 1, the greater the similarity value. However, in this version of the code, 
    # the cosine distance is subtracted by 1 and multiplied by 1000, initially to better observe the difference between similarities.
    # At this point in the code, the smaller the value in the similarity matrix, the more similar it is.  
    # You can operate the similarity matrix as follows: similarity_matrix = 1 - similarity_matrix/1000;  
    # And change the mapping interval to: Mapping_ranges = [(2, 1.8),(1.8, 1.6),(1.6, 1.4),(1.4, 1.2),(1.2, 1), 
    # (1, 0.8),(0.8, 0.6),(0.6, 0.4),(0.4, 0.2),(0.2, 0.01)].  
    # This corresponds to the explanation in the paper. It's the same in theory.
    # The two are only different in operation, and there is no essential difference.
    
    similarity_matrix = np.zeros_like(image)
    for i in range(image.shape[1]):
        for j in range(image.shape[2]):
            center_pixel = image[:, i, j]
            neighbors = []
            # Get the values of the four surrounding pixels
            if i > 0:
                neighbors.append(image[:, i - 1, j])
            if i < image.shape[1] - 1:
                neighbors.append(image[:, i + 1, j])
            if j > 0:
                neighbors.append(image[:, i, j - 1])
            if j < image.shape[2] - 1:
                neighbors.append(image[:, i, j + 1])
            a = np.array(neighbors)
            if neighbors:
                similarities = cosine_similarity(center_pixel.reshape(1, -1), np.array(neighbors))
                similarity_matrix[:, i, j] = (1 - np.mean(similarities)) * 1000
    if dataset == 'Samson':
        sio.savemat('similarity_samson.mat', {'samson': similarity_matrix})
    elif dataset == 'Urban':
        sio.savemat('similarity_urban.mat', {'urban': similarity_matrix})
    elif dataset == 'dc':
        sio.savemat('similarity_dc.mat', {'dc': similarity_matrix})
    """

    if dataset == 'Samson':
        similarity_matrix = sio.loadmat('similarity_samson.mat')['samson']
    elif dataset == 'Urban':
        similarity_matrix = sio.loadmat('similarity_urban.mat')['urban']
    elif dataset == 'dc':
        similarity_matrix = sio.loadmat('similarity_dc.mat')['dc']

    similarity_matrix = similarity_matrix[0]
    thre = threshold_otsu(similarity_matrix)

    flattened_matrix = similarity_matrix.flatten()
    mapping_ranges = [(0.01, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1), (1, 1.2), (1.2, 1.4), (1.4, 1.6),
                      (1.6, 1.8), (1.8, 2)]
    mask = binning_mapping(similarity_matrix, thre, mapping_ranges)

    mask = torch.from_numpy(mask).unsqueeze(0).to(device)

    # load data
    train_dataset = load_data(img=original_HSI, transform=transforms.ToTensor())
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    net = Unmixing(band_number, endmember_number, drop_out, col).cuda()

    endmember_name = datasetnames[dataset] + '_run' + str(run)
    endmember_path = endmember_folder + '/' + endmember_name
    endmember_path2 = endmember_folder + '/' + endmember_name + 'vca'

    abundance_name = datasetnames[dataset] + '_run' + str(run)
    abundance_path = abundance_folder + '/' + abundance_name


    def weights_init(m):
        nn.init.kaiming_normal_(net.layer1[0].weight.data)
        nn.init.kaiming_normal_(net.layer1[4].weight.data)
        nn.init.kaiming_normal_(net.layer1[8].weight.data)


    net.apply(weights_init)

    # decoder weight init by VCA
    model_dict = net.state_dict()
    model_dict["decoderlayer4.0.weight"] = endmember_init

    net.load_state_dict(model_dict)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in range(EPOCH):
        for i, x in enumerate(train_loader):
            x = x.cuda()
            net.train().cuda()

            en_abundance, reconstruction_result = net(x, mask)

            abundanceLoss = reconstruction_SADloss(x, reconstruction_result)
            total_loss = abundanceLoss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 100 == 0:
                """print(ELoss.cpu().data.numpy())"""
                print("Epoch:", epoch, "| loss: %.4f" % total_loss.cpu().data.numpy())

    net.eval()

    en_abundance, reconstruction_result = net(x, mask)
    en_abundance = torch.squeeze(en_abundance)

    en_abundance = torch.reshape(en_abundance, [endmember_number, col * col])
    en_abundance = en_abundance.T
    en_abundance = torch.reshape(en_abundance, [col, col, endmember_number])
    abundance_GT = torch.reshape(abundance_GT, [endmember_number, col * col])
    abundance_GT = abundance_GT.T
    abundance_GT = torch.reshape(abundance_GT, [col, col, endmember_number])
    en_abundance = en_abundance.cpu().detach().numpy()
    abundance_GT = abundance_GT.cpu().detach().numpy()

    endmember_hat = net.state_dict()["decoderlayer4.0.weight"].cpu().numpy()
    endmember_hat = np.squeeze(endmember_hat)
    endmember_hat = endmember_hat.T

    GT_endmember = GT_endmember.T
    y_hat = reconstruct(en_abundance, endmember_hat)
    RE = np.sqrt(np.mean(np.mean((y_hat - data) ** 2, axis=1)))
    r.append(RE)

    sio.savemat(mat_folder + '/' + method_name + '_run' + str(run) + '.mat', {'A': en_abundance,
                                                                              'E': endmember_hat, })

    plotAbundancesSimple(en_abundance, abundance_GT, abundance_path, abu)
    plotEndmembersAndGT(endmember_hat, GT_endmember, endmember_path, end)

    torch.cuda.empty_cache()

    print('-' * 70)
end_time = time.time()
end = np.reshape(end, (-1, endmember_number + 1))
abu = np.reshape(abu, (-1, endmember_number + 1))
dt = pd.DataFrame(end)
dt2 = pd.DataFrame(abu)
dt3 = pd.DataFrame(r)
# SAD and mSAD results of each endmember for multiple runs are saved to csv files
dt.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' +
          'SAD and mSAD results for multiple runs.csv')
# RMSE and mRMSE results of each abundance for multiple runs are saved to csv files
dt2.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' +
           'RMSE and mRMSE results for multiple runs.csv')
# RE results of each abundance for multiple runs are saved to csv files
dt3.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'RE results for multiple runs.csv')
# abundance GT
abundanceGT_path = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'Abundance GT'
plotAbundancesGT(hsi.abundance_gt, abundanceGT_path)
print('Running time:', end_time - start_time, 's')
