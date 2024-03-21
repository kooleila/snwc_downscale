FROM rockylinux/rockylinux:8

RUN rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm \
             https://download.fmi.fi/smartmet-open/rhel/8/x86_64/smartmet-open-release-latest-8.noarch.rpm

RUN dnf -y install dnf-plugins-core && \
    dnf config-manager --set-enabled powertools && \
    dnf config-manager --setopt="epel.exclude=eccodes*" --save && \
    dnf -y --setopt=install_weak_deps=False install python39 python39-pip python39-setuptools eccodes git && \
    dnf -y install \
        hdf5-devel \
        #netcdf-bin \
        netcdf-devel \
        gcc \
        make \
        libtool \
    && dnf -y clean all && \
    rm -rf /var/cache/dnf

RUN git clone https://github.com/kooleila/snwc_downscale.git

WORKDIR /snwc_downscale

ADD https://lake.fmi.fi/dem-data/DEM_100m-Int16.tif /snwc_downscale
ADD https://lake.fmi.fi/ml-data/elev_100m_1000m.nc /snwc_downscale
ADD https://lake.fmi.fi/ml-data/maa_meri_lcc_1000.nc /snwc_downscale
ADD https://lake.fmi.fi/ml-models/mnwc-biascorrection/xgb_T2m_1023.joblib /snwc_downscale
ADD https://lake.fmi.fi/ml-models/mnwc-biascorrection/xgb_WS_1023.joblib /snwc_downscale
ADD https://lake.fmi.fi/ml-models/mnwc-biascorrection/xgb_WG_1023.joblib /snwc_downscale
ADD https://lake.fmi.fi/ml-models/mnwc-biascorrection/xgb_RH_1023.joblib /snwc_downscale

RUN chmod 644 DEM_100m-Int16.tif && \
    chmod 644 elev_100m_1000m.nc && \
    chmod 644 maa_meri_lcc_1000.nc && \
    chmod 644 xgb_T2m_1023.joblib && \
    chmod 644 xgb_WS_1023.joblib && \
    chmod 644 xgb_WG_1023.joblib && \
    chmod 644 xgb_RH_1023.joblib && \
    update-alternatives --set python3 /usr/bin/python3.9 && \
    python3 -m pip --no-cache-dir install -r requirements.txt


