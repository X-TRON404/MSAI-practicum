import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2


from utils import load_sample, run_gradcam, visualize_slice

class CNN3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool3 = nn.MaxPool3d(kernel_size=2)
        
        self.conv4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3)
        self.pool4 = nn.MaxPool3d(kernel_size=2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(in_features=256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = F.relu(self.pool4(self.conv4(x)))
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = CNN3D()
model.load_state_dict(torch.load('./pneumonitis-cnn-dn2.pt'))
model.eval()
model.cuda()

uploaded_file_lung = st.file_uploader("Please upload lung segmentation mask", type="npy")
uploaded_file_dose = st.file_uploader("Please upload dose pattern", type="npy")
if uploaded_file_lung is not None and uploaded_file_dose is not None:
    lung, dose = load_sample(uploaded_file_lung, uploaded_file_dose)
    st.text("loading finished")
    
    fig = visualize_slice(lung, dose)
    st.pyplot(fig)
    
    lung_tensor = torch.from_numpy(lung.transpose((2, 0, 1))).float().unsqueeze(0)
    dose_tensor = torch.from_numpy(dose.transpose((2, 0, 1))).float().unsqueeze(0)
    inputs = lung_tensor * dose_tensor

    st.text("running model ...")
    with torch.no_grad():
        pred = model(inputs.cuda().unsqueeze(0))

    st.write(f"Possibility of pneumonitis: {pred.item()*100:.2f}%")

    if pred.item() > 0.57:
        st.write("Result: Positive")
    else:
        st.write("Result: Negative")

    if st.button("GradCAM"):
        st.text("running GradCAM ...")
        hm_conv4, hm_conv3 = run_gradcam(model, lung, dose)
        hm_conv3 = np.random.random((128, 128))  # for test only
        fig, ax = plt.subplots()
        ax.imshow(hm_conv3, cmap='jet')
        st.pyplot(fig)

        # # 可视化 Heatmap
        # plt.imshow(grayscale_cam, cmap='jet')
        # plt.colorbar()
        # plt.title('Dummy Heatmap')
        # plt.axis('off')

        # # 是否叠加 Heatmap
        # overlay_option = st.checkbox("将 Heatmap 叠加在输入上")

        # if overlay_option:
        #     input_image = data.cpu().numpy()[0].transpose(1, 2, 0)
        #     input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
        #     heatmap = np.uint8(255 * grayscale_cam)
        #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #     superimposed_img = heatmap * 0.4 + input_image * 0.6
        #     st.image(superimposed_img, caption='Heatmap Overlay', use_column_width=True)
        # else:
        #     st.pyplot(plt)