FROM arcaneframework/alien-base:ubuntu20.04
LABEL maintener.email=<cedric.chevalier@cea.fr>

ARG UID=1000
ARG GID=100

RUN addgroup --gid $GID user \
    && adduser --gid $GID --uid $UID user \
    && yes ubuntu2004 | passwd user

# enable clion support
# see https://github.com/JetBrains/clion-remote
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    openssh-server \
    gdb valgrind \
    && apt-get clean

RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
    ) > /etc/ssh/sshd_config_test_clion \
    && mkdir /run/sshd

CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]