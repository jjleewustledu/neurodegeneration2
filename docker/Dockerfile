# See also: https://hub.docker.com/r/tensorflow/tensorflow and
#           https://hub.docker.com/r/tensorflow/tensorflow/tags/?page=1&name=2.3.3
# The OS is Ubuntu 18.04.  nvidia-docker is needed to run Nvidia CUDA 10+.  A jupyter notebook server starts on boot.
# Tensorflow tutorial notebooks are included.  Mount a volumeto /tf/notebooks for work with your own notebooks.
FROM tensorflow/tensorflow:2.3.3-gpu-jupyter

# See also a useful Docker tutorial with links:
# https://towardsdatascience.com/how-docker-can-help-you-become-a-more-effective-data-scientist-7fc048ef91d5
# Do ask google.
LABEL maintainer="John J. Lee <www.github.com/jjleewustledu>"

# Define OS environment
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV	DEBIAN_FRONTEND=noninteractive
RUN	ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime

# Build the Docker image with required and desired Ubuntu packages.
RUN apt-get update --fix-missing && apt-get install -y \
    apt-utils \
    wget \
    curl \
    bzip2 \
    unzip \
    ca-certificates \
    build-essential \
    git-core \
    pkg-config \
    python-pip \
    python-setuptools \
    python-virtualenv \
    python3-tk \
    tree \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Define working environment
RUN mkdir /uufs /data
ENV HOME=/uufs
ENV SHELL=/bin/bash
VOLUME /data

# Ensure that TensorFlow can be imported and session started (session start touches GPU)
RUN	python -c "import tensorflow as tf;s = tf.constant( 1.0, tf.float32 )"

# Build DeepMRSeg.  See also the forked repository.
#RUN pip --no-cache-dir install --upgrade setuptools
RUN	mkdir -p /opt/cbica/src/deepmrseg && \
	cd /opt/cbica/src/deepmrseg && \
	git clone https://github.com/CBICA/DeepMRSeg.git && \
	cd DeepMRSeg && \
	python setup.py install && \
	rm -rf /opt/cbica/src/deepmrseg && \
	rmdir /opt/cbica/src /opt/cbica
RUN deepmrseg_downloadmodel --model dlicv && \
    tree ~/.deepmrseg
WORKDIR /data
ENTRYPOINT ["deepmrseg_apply"]
#ENTRYPOINT ["/bin/bash"]



# Build Docker image
# ------------------
# "% nvidia-docker build -t jjleewustledu/deepmrseg_image:20220615 -f ./Dockerfile ./"
#
# Implement a quick application.
# ------------------------------
# "% nvidia-docker run -it -v $(pwd):/data --rm jjleewustledu/deepmrseg_image:20220615 --task dlicv --inImg fileprefix.nii.gz --outImg fileprefix_DLICV.nii.gz"
#
# will invoke the following from the within the container
# "tf-docker /data > deepmrseg_apply --task dlicv --inImg fileprefix.nii.gz --outImg fileprefix_DLICV.nii.gz"
# See also:  https://stackoverflow.com/questions/53543881/docker-run-pass-arguments-to-entrypoint
#
# Implement production application.
# ---------------------------------
# "% nvidia-docker run -it -v $(pwd):/data --rm deepmrseg_image:20220615 --task dlicv --sList SLIST"
# SLIST := Image list file name. Enter a comma separated list file with
#          columns for: ID, input image(s) and output image. (REQUIRED)
#
# Push tested application.
# ------------------------
# "% nvidia-docker images # to see local images"
# "% nvidia-docker push jjleewustledu/deepmrseg_image:20220615"
#
# Implement with singularity.
# ---------------------------
# KLUDGE:  ensure that ~/.deepmrseg/trained_models/dlicv/DeepMRSeg_DLICV_v1.0/ exists on singularity cluster and is populated with models.  Container may expect to find .deepmrseg in unexpected locations; accomodate container's error messages.
# "% singularity pull docker://jjleewustledu/deepmrseg_image:20220615"
# '% singularity exec --bind $(pwd):/data /home/aris_data/ADNI_FDG/bin/deepmrseg_image_20220615.sif "deepmrseg_apply" "--task" "dlicv" "--inImg" "/data/fileprefix.nii.gz" "--outImg" "/data/fileprefix_DLICV.nii.gz"' ### no gpu
# '% singularity exec --nv --bind $(pwd):/data /home/aris_data/ADNI_FDG/bin/deepmrseg_image_20220615.sif "deepmrseg_apply" "--task" "dlicv" "--inImg" "/data/fileprefix.nii.gz" "--outImg" "/data/fileprefix_DLICV.nii.gz"' ### with gpu
#
# '% singularity exec --nv --bind $(pwd):/data /home/aris_data/ADNI_FDG/bin/deepmrseg_image_20220615.sif "bash"' ### debug



# More hints from https://hub.docker.com/r/tensorflow/tensorflow
# --------------------------------------------------------------
# $ docker run -it --rm tensorflow/tensorflow bash
# Start a CPU-only container
#
# $ docker run -it --rm --runtime=nvidia tensorflow/tensorflow:latest-gpu python
# Start a GPU container, using the Python interpreter.
#
# $ docker run -it --rm -v $(realpath ~/notebooks):/tf/notebooks -p 8888:8888 tensorflow/tensorflow:latest-jupyter
# Run a Jupyter notebook server with your own notebook directory (assumed here to be ~/notebooks).
# To use it, navigate to localhost:8888 in your browser.