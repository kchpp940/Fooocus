FROM nvidia/cuda:12.4.1-base-ubuntu22.04
ENV DEBIAN_FRONTEND noninteractive
ENV CMDARGS --listen

ENV FOOOCUS_DATA_DIR /data
ENV FOOOCUS_USER_DATA_DIR /data
ENV DATADIR /data

RUN apt-get update -y && \
	apt-get install -y curl libgl1 libglib2.0-0 python3-pip python-is-python3 git && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

COPY requirements_docker.txt requirements_versions.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements_docker.txt -r /tmp/requirements_versions.txt && \
	rm -f /tmp/requirements_docker.txt /tmp/requirements_versions.txt
RUN pip install --no-cache-dir xformers==0.0.23 --no-dependencies
RUN curl -fsL -o /usr/local/lib/python3.10/dist-packages/gradio/frpc_linux_amd64_v0.2 https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_amd64 && \
	chmod +x /usr/local/lib/python3.10/dist-packages/gradio/frpc_linux_amd64_v0.2

RUN adduser --disabled-password --gecos '' user

RUN mkdir -p /app /data/models /data/cache /data/outputs /data/config /data/user_presets

COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh && \
	chown -R user:user /app /data

WORKDIR /app
USER user

COPY --chown=user:user . /app

ENV config_path=/data/config.txt
ENV config_example_path=/data/config_modification_tutorial.txt
ENV sorted_styles_path=/data/sorted_styles.json
ENV path_checkpoints=/data/models/checkpoints/
ENV path_loras=/data/models/loras/
ENV path_embeddings=/data/models/embeddings/
ENV path_vae=/data/models/vae/
ENV path_vae_approx=/data/models/vae_approx/
ENV path_upscale_models=/data/models/upscale_models/
ENV path_inpaint=/data/models/inpaint/
ENV path_controlnet=/data/models/controlnet/
ENV path_clip_vision=/data/models/clip_vision/
ENV path_fooocus_expansion=/data/models/prompt_expansion/fooocus_expansion/
ENV path_safety_checker=/data/models/safety_checker/
ENV path_sam=/data/models/sam/
ENV path_outputs=/data/outputs/
ENV temp_path=/data/cache/temp/

VOLUME ["/data"]

CMD [ "sh", "-c", "/entrypoint.sh ${CMDARGS}" ]
