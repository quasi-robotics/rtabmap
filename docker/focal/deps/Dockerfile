# Image: introlab3it/rtabmap:focal

FROM ros:noetic-perception

ARG TARGETPLATFORM
ENV TARGETPLATFORM=${TARGETPLATFORM:-linux/amd64}
RUN echo "I am building for $TARGETPLATFORM"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /root/

# issue: https://github.com/introlab/rtabmap/issues/1523
RUN rm /etc/apt/sources.list.d/ros1-latest.list || true && \
    apt-get update && apt-get install -y curl && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - && \
    apt-get clean && rm -rf /var/lib/apt/lists/

# Install build dependencies
RUN if [ "$TARGETPLATFORM" = "linux/arm/v7" ]; then \
    apt-get update && \
    apt-get install -y git software-properties-common libtbb-dev zlib1g-dev liboctomap-dev ros-noetic-visualization-msgs ros-noetic-velodyne-pointcloud ros-noetic-turtlebot3-navigation ros-noetic-turtlebot3-bringup ros-noetic-tf-conversions ros-noetic-tf2-ros ros-noetic-tf ros-noetic-stereo-msgs ros-noetic-std-srvs ros-noetic-std-msgs ros-noetic-sensor-msgs ros-noetic-rviz ros-noetic-rospy ros-noetic-roscpp ros-noetic-robot-localization ros-noetic-realsense2-camera qtbase5-dev libproj-dev ros-noetic-pluginlib ros-noetic-pcl-ros ros-noetic-pcl-conversions ros-noetic-octomap-msgs ros-noetic-nodelet ros-noetic-nav-msgs ros-noetic-move-base-msgs ros-noetic-message-runtime ros-noetic-message-generation ros-noetic-message-filters libsqlite3-dev libqt5widgets5 libqt5core5a libpcl-dev ros-noetic-libg2o ros-noetic-laser-geometry ros-noetic-imu-filter-madgwick ros-noetic-image-transport ros-noetic-image-geometry ros-noetic-husky-navigation ros-noetic-hector-mapping ros-noetic-gtsam ros-noetic-grid-map-ros ros-noetic-grid-map-core ros-noetic-geometry-msgs ros-noetic-genmsg ros-noetic-find-object-2d ros-noetic-eigen-conversions ros-noetic-dynamic-reconfigure ros-noetic-diagnostic-updater ros-noetic-cv-bridge ros-noetic-costmap-2d ros-noetic-catkin ros-noetic-apriltag-ros ros-noetic-actionlib-msgs ros-noetic-actionlib wget libfreenect-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/ ;fi
RUN if [ "$TARGETPLATFORM" != "linux/arm/v7" ]; then \
    apt-get update && \
    apt-get install -y git software-properties-common libtbb-dev libfreenect-dev ros-noetic-rtabmap-ros && \
    apt-get remove -y ros-noetic-rtabmap* && \
    apt-get clean && rm -rf /var/lib/apt/lists/ ;fi

#https://gitlab.kitware.com/cmake/cmake/-/issues/20568
RUN if [ "$TARGETPLATFORM" = "linux/arm/v7" ]; then \
    wget -nv https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz   && \
    tar -xzf cmake-3.20.0.tar.gz  && \
    cd cmake-3.20.0 && \
    export CFLAGS="-D_FILE_OFFSET_BITS=64" && \
    export CXXFLAGS="-D_FILE_OFFSET_BITS=64" && \
    ./bootstrap && \
    make -j4 && \
    sudo make install && \
    cd .. && \
    cmake --version && \
    rm -r cmake-3.20.0.tar.gz cmake-3.20.0 ;fi
 
# MRPT
RUN add-apt-repository ppa:joseluisblancoc/mrpt-stable -y
RUN apt-get update && apt install libmrpt-poses-dev -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/

# PDAL
RUN apt-get update && apt-get install -y libpdal-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/

# RealSense2
RUN apt-get update && apt-get install -y ros-noetic-librealsense2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/

# Azure Kinect DK
# Taken from https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/1190#issuecomment-822772494
# K4A binaries on 20.04 not released yet, we should take those from 18.04
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then echo "Installing k4a..." && \
    apt-get update && \
    echo "Download libk4a1.3_1.3.0_amd64.deb..." && \
    curl -sSL https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.3/libk4a1.3_1.3.0_amd64.deb > /tmp/libk4a1.3_1.3.0_amd64.deb && \
    echo "Download libk4a1.3-dev_1.3.0_amd64.deb..." && \
    curl -sSL https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.3-dev/libk4a1.3-dev_1.3.0_amd64.deb > /tmp/libk4a1.3-dev_1.3.0_amd64.deb && \
    echo "Download libk4abt1.0_1.0.0_amd64.deb..." && \
    curl -sSL https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4abt1.0/libk4abt1.0_1.0.0_amd64.deb > /tmp/libk4abt1.0_1.0.0_amd64.deb && \
    echo "Download libk4abt1.0-dev_1.0.0_amd64.deb..." && \
    curl -sSL https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4abt1.0-dev/libk4abt1.0-dev_1.0.0_amd64.deb > /tmp/libk4abt1.0-dev_1.0.0_amd64.deb && \
    echo "Download k4a-tools_1.3.0_amd64.deb..." && \
    curl -sSL https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/k/k4a-tools/k4a-tools_1.3.0_amd64.deb > /tmp/k4a-tools_1.3.0_amd64.deb && \
    echo "Accept license..." && \
    echo 'libk4a1.3 libk4a1.3/accepted-eula-hash string 0f5d5c5de396e4fee4c0753a21fee0c1ed726cf0316204edda484f08cb266d76' | debconf-set-selections && \
    echo 'libk4abt1.0	libk4abt1.0/accepted-eula-hash	string	03a13b63730639eeb6626d24fd45cf25131ee8e8e0df3f1b63f552269b176e38' | debconf-set-selections && \
    dpkg -i /tmp/libk4a1.3_1.3.0_amd64.deb && \
    dpkg -i /tmp/libk4a1.3-dev_1.3.0_amd64.deb && \
    dpkg -i /tmp/libk4abt1.0_1.0.0_amd64.deb && \
    dpkg -i /tmp/libk4abt1.0-dev_1.0.0_amd64.deb && \
    apt-get install -y libsoundio1 && \
    dpkg -i /tmp/k4a-tools_1.3.0_amd64.deb && \
    rm /tmp/libk4a* /tmp/k4a* && \
    apt-get clean && rm -rf /var/lib/apt/lists/; fi

# libfreenect2 
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then echo "Installing libfreenect2..." && \
    apt-get update && apt-get install -y mesa-utils xserver-xorg-video-all libusb-1.0-0-dev libturbojpeg0-dev libglfw3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/ && \
    git clone https://github.com/OpenKinect/libfreenect2 && \
    cd libfreenect2 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j4 && \
    make install && \
    cd && \
    rm -r libfreenect2; fi

# zed open capture
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then echo "Installing zed-open-capture..." && \
    apt-get update && apt install libusb-1.0-0-dev libhidapi-libusb0 libhidapi-dev wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/ && \
    git clone https://github.com/stereolabs/zed-open-capture.git && \
    cd zed-open-capture && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j4 && \
    make install && \
    cd && \
    rm -r zed-open-capture; fi

# AliceVision v2.4.0 modified (Sept 13 2021)
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then echo "Installing AliceVision..." && \
      apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      libsuitesparse-dev \
      libceres-dev \
      xorg-dev \
      libglu1-mesa-dev \
      wget && apt-get clean && rm -rf /var/lib/apt/lists/; fi
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then git clone https://github.com/OpenImageIO/oiio.git && \
    cd oiio && \
    git checkout Release-2.0.12 && \
    mkdir build && \
    cd build && \
    cmake -DUSE_PYTHON=OFF -DOIIO_BUILD_TESTS=OFF -DOIIO_BUILD_TOOLS=OFF .. && \
    make -j4 && \
    make install && \
    cd && \
    rm -r oiio; fi
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then git clone https://github.com/assimp/assimp.git && \
    cd assimp && \
    git checkout 71a87b653cd4b5671104fe49e2e38cf5dd4d8675 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j4 && \
    make install && \
    cd && \
    rm -r assimp; fi
# Ignore assimp installed from ros, as it has invalid "/usr/lib/include" in its target, making rtabmap's cmake fails
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then rm -rf /usr/lib/x86_64-linux-gnu/cmake/assimp*; fi
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then git clone https://github.com/alicevision/geogram.git && \
    cd geogram && \
    git checkout v1.7.6 && \
    wget https://gist.githubusercontent.com/matlabbe/1df724465106c056ca4cc195c81d8cf0/raw/b3ed4cb8f9b270833a40d57d870a259eabfa4415/geogram_8b2ae61.patch && \
    git apply geogram_8b2ae61.patch && \
    ./configure.sh && \
    cd build/Linux64-gcc-dynamic-Release && \
    make -j4 && \
    make install && \
    cd && \
    rm -r geogram; fi
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then git clone https://github.com/alicevision/AliceVision.git --recursive && \
    cd AliceVision && \
    git checkout 0f6115b6af6183c524aa7fcf26141337c1cf3872 && \
    git submodule update -i && \
    wget https://gist.githubusercontent.com/matlabbe/1df724465106c056ca4cc195c81d8cf0/raw/5e3437cf6229c8d534bbaee475ed9bcb92eb84a1/alicevision_0f6115b.patch && \
    git apply alicevision_0f6115b.patch && \
    mkdir build && \
    cd build && \
    cmake -DALICEVISION_USE_CUDA=OFF -DALICEVISION_USE_APRILTAG=OFF -DALICEVISION_BUILD_SOFTWARE=OFF .. && \
    make -j4 && \
    make install && \
    cd && \
    rm -r AliceVision; fi
    
RUN git clone --branch 4.2.0 https://github.com/opencv/opencv.git && \
    git clone --branch 4.2.0 https://github.com/opencv/opencv_contrib.git && \
    cd opencv && \
    mkdir build && \
    cd build && \
    cmake -DWITH_TBB=ON -DWITH_OPENMP=ON -DBUILD_opencv_python3=OFF -DBUILD_opencv_python_bindings_generator=OFF -DBUILD_opencv_python_tests=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DOPENCV_ENABLE_NONFREE=ON -DWITH_VTK=OFF -DOPENCV_EXTRA_MODULES_PATH=/root/opencv_contrib/modules .. && \
    make -j4 && \
    make install && \
    cd ../.. && \
    rm -rf opencv opencv_contrib

RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then apt-get update && apt install wget && apt-get clean && rm -rf /var/lib/apt/lists/; fi
RUN git clone https://github.com/laurentkneip/opengv.git && \
    cd opengv && \
    git checkout 91f4b19c73450833a40e463ad3648aae80b3a7f3 && \
    wget https://gist.githubusercontent.com/matlabbe/a412cf7c4627253874f81a00745a7fbb/raw/accc3acf465d1ffd0304a46b17741f62d4d354ef/opengv_disable_march_native.patch && \
    git apply opengv_disable_march_native.patch && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j4 && \
    make install && \
    cd && \
    rm -r opengv

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# for jetson (https://github.com/introlab/rtabmap/issues/776)
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/tegra
