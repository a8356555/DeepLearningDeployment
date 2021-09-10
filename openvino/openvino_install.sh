
# !lscpu # 檢查CPU狀態
# !cat /etc/os-release # 檢查作業系統狀態

# # 1. 安裝Intel OpenVINO工具包

# 取得OpenVINO2020公開金錀
wget https://apt.repos.intel.com/openvino/2021/GPG-PUB-KEY-INTEL-OPENVINO-2021
# 加入OpenVINO公開金錀到系統金錀群中
sudo apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2021
# 建立更新安裝清單檔案
sudo touch /etc/apt/sources.list.d/intel-openvino-2021.list
# 將下載指令加入安裝清單中
echo "deb https://apt.repos.intel.com/openvino/2021 all main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2021.list
# 更新系統
sudo apt update
sudo apt-cache search intel-openvino-dev-ubuntu18

# 安裝OpenVINO到虛擬機系統中
sudo apt install -y intel-openvino-dev-ubuntu18-2021.1.110 
# 列出安裝路徑下內容進行確認
!ls /opt/intel
%cd /opt/intel/openvino_2021/install_dependencies
!sudo -E ./install_openvino_dependencies.sh
!source /opt/intel/openvino_2021/bin/setupvars.sh
%cd /opt/intel/openvino_2021/deployment_tools/model_optimizer/install_prerequisites
!sudo ./install_prerequisites.sh
!pip install openvino
# export LD_LIBRARY_PATH=/opt/intel/openvino_2021/inference_engine/lib/intel64/:$LD_LIBRARY_PATH




# # - - - - - - - - - - - - 
# # 2. 模型下載
# # 執行環境設定批次檔並以模型下載器取得mobilenet-v1-1.0-224
# !source /opt/intel/openvino_2021/bin/setupvars.sh && \
# python /opt/intel/openvino_2021/deployment_tools/tools/model_downloader/downloader.py \
# --name mobilenet-v1-1.0-224

# # 3. 模型優化
# # 下載及安裝test-generator 方便檢查程式運行錯誤
# !pip install test-generator==0.1.1

# # 執行環境設定批次檔並將下載到的mobilenet-v1-1.0-224模型檔進行優化轉換產生IR(xml & bin)檔
# !source /opt/intel/openvino_2021/bin/setupvars.sh && \
# python /opt/intel/openvino_2021/deployment_tools/tools/model_downloader/converter.py \
# --name mobilenet-v1-1.0-224