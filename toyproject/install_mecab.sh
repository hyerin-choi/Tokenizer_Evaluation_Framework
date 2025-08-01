#!/bin/bash

# MeCab과 한국어 사전 설치 스크립트
# Ubuntu/Debian/CentOS/RHEL 시스템용

echo "=== MeCab과 한국어 사전 설치 시작 ==="

# 시스템 업데이트
echo "1. 시스템 패키지 업데이트 중..."
if command -v apt-get &> /dev/null; then
    # Ubuntu/Debian
    sudo apt-get update
    sudo apt-get install -y build-essential libmecab-dev mecab-ipadic-utf8
elif command -v yum &> /dev/null; then
    # CentOS/RHEL
    sudo yum update -y
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y mecab mecab-ipadic
else
    echo "지원되지 않는 패키지 매니저입니다. 수동으로 설치해주세요."
    exit 1
fi

# MeCab 소스코드 다운로드 및 설치
echo "2. MeCab 소스코드 설치 중..."
cd /tmp
wget https://github.com/konlpy/mecab-ko-dic/archive/refs/heads/master.zip
unzip master.zip
cd mecab-ko-dic-master

# 한국어 사전 컴파일
echo "3. 한국어 사전 컴파일 중..."
./configure
make
sudo make install

# 사전 경로 설정
echo "4. 사전 경로 설정 중..."
sudo mkdir -p /usr/local/lib/mecab/dic/mecab-ko-dic
sudo cp -r * /usr/local/lib/mecab/dic/mecab-ko-dic/

# 환경 변수 설정
echo "5. 환경 변수 설정 중..."
echo 'export MECABRC="/usr/local/etc/mecabrc"' >> ~/.bashrc
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Python 패키지 설치
echo "6. Python 패키지 설치 중..."
pip install mecab-python3 konlpy

echo "=== MeCab 설치 완료 ==="
echo "설치된 사전 경로: /usr/local/lib/mecab/dic/mecab-ko-dic"
echo "환경 변수를 적용하려면 터미널을 재시작하거나 'source ~/.bashrc'를 실행하세요." 