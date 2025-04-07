#!/bin/bash

# Parse command line arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      echo "Usage: $0 [options] <download_dir>"
      echo "Options:"
      echo "\t -a \t Short-hand for -d -u -r."
      echo "\t -d \t Download the dataset."
      echo "\t -v \t Verify archive integrity."
      echo "\t -u \t Unpack the dataset."
      echo "\t -r \t Remove the downloaded archives."
      echo "\t -h \t Show this help."
      exit 0
      ;;
    -a)
      DOWNLOAD=true
      UNPACK=true
      REMOVE_ARCHIVES=true
      shift # past argument
      ;;
    -d)
      DOWNLOAD=true
      shift # past argument
      ;;
    -v)
      VERIFY=true
      shift # past argument
      ;;
    -u)
      UNPACK=true
      shift # past argument
      ;;
    -r)
      REMOVE_ARCHIVES=true
      shift # past argument
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done
set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters
DATASET_DIR="$1"
EXEC_DIR=$(pwd)

echo "MVPSP Dataset Downloader"
echo "Download: ${DOWNLOAD-false}"
echo "Verify: ${VERIFY-false}"
echo "Unpack: ${UNPACK-false}"
echo "Remove archives: ${REMOVE_ARCHIVES-false}"
echo "Dataset location: ${DATASET_DIR}"

mkdir -p "${DATASET_DIR}" && cd "${DATASET_DIR}"
retVal=$?
if [ ${retVal} -ne 0 ]; then
  echo "Could not create dataset directory: Error ${retVal}"
  # cd back into previous location
  cd "${EXEC_DIR}"
  exit ${retVal};
fi

# download dataset
if [[ -v DOWNLOAD ]]; then
  # Check if azcopy is installed
  export PATH="${PATH:+${PATH}:}$(pwd)"
  AZCOPY="$(which azcopy)"
  if [ -z "${AZCOPY}" ]; then
    echo "Error: Could not find azcopy utility. Please check out https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10 to install it.";
    exit 1;
  fi
  echo "Downloading MVPSP dataset to ${DATASET_DIR}..."
  ${AZCOPY} copy --recursive --overwrite=false 'https://rocs2.blob.core.windows.net/mvpsp-public-mirror/*?sv=2022-11-02&ss=bfqt&srt=sco&sp=rlx&se=2029-07-17T17:39:08Z&st=2024-07-17T09:39:08Z&spr=https&sig=cITpUGxTo1aNHvFaMysP%2BCS%2Fy9OHfvXiwuoXGTSlBC4%3D' .
  retVal=$?
  if [ ${retVal} -ne 0 ]; then
    echo "Download error: azcopy returned ${retVal}"
    # cd back into previous location
    cd "${EXEC_DIR}"
    exit ${retVal};
  fi
  echo "Download complete."
fi

# verify dataset
if [[ -v VERIFY ]]; then
  echo "Verifying archive integrity..."
  find . -maxdepth 1 -type f -iname "mvpsp.tar.xz" -or -iname "test_orx_*.tar.xz" -or -iname "test_wetlab_*.tar.xz" -or -iname "train_wetlab_*.tar.xz" -or -iname "train_synth.tar.xz" -or -iname "train_pbr.tar.xz" | parallel "echo {} ; tar -tf {} &> /dev/null || echo Archive failed integrity check: {}"
  retVal=$?
  if [ ${retVal} -ne 0 ]; then
    echo "Verification error: tar returned ${retVal}"
    # cd back into previous location
    cd "${EXEC_DIR}"
    exit ${retVal};
  fi
  echo "Integrity check complete."
fi

# unpack dataset
if [[ -v UNPACK ]]; then
  echo "Unpacking dataset..."
  COMMAND="echo {} ; tar -xJf {}"
  if [[ -v REMOVE_ARCHIVES ]]; then
    COMMAND="${COMMAND} && rm {}"
  fi
  find . -maxdepth 1 -type f -iname "mvpsp.tar.xz" -or -iname "test_orx_*.tar.xz" -or -iname "test_wetlab_*.tar.xz" -or -iname "train_wetlab_*.tar.xz" -or -iname "train_synth.tar.xz" -or -iname "train_pbr.tar.xz" | parallel "${COMMAND}"
  retVal=$?
  if [ ${retVal} -ne 0 ]; then
    echo "Unpack error: tar returned ${retVal}"
    # cd back into previous location
    cd "${EXEC_DIR}"
    exit ${retVal};
  fi
  echo "Unpacking done."
fi

# remove dataset archives
if ! [[ -v UNPACK ]] && [[ -v REMOVE_ARCHIVES ]]; then
  echo "Cleaning up archives..."
  find . -maxdepth 1 -type f -iname "mvpsp.tar.xz" -or -iname "test_orx_*.tar.xz" -or -iname "test_wetlab_*.tar.xz" -or -iname "train_wetlab_*.tar.xz" -or -iname "train_synth.tar.xz" -or -iname "train_pbr.tar.xz" -print -exec rm {} +
  echo "Clean up done."
fi

# cd back into previous location
cd "${EXEC_DIR}"
