FROM stefco/hpmoc-env
COPY dist dist
RUN pip install dist/hpmoc-latest-py3-none-any.whl \
    && rm -rf ~/.cache \
    && conda clean -y --all \
    && rm -r dist \
