#!/bin/bash

# Parse command line arguments
POSITIONAL_ARGS=()
SUBSETS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      echo "Usage: $0 [options] <download_dir>"
      echo "Options:"
      echo -e " -a \t\t Short-hand for -d -u -r."
      echo -e " -d [<subsets>]\t Download the dataset. Optionally takes a list of space-separated subset names to download. Valid subset names are train_wetlab train_synth train_pbr test_wetlab test_orx ."
      echo -e " -v \t\t Verify archive integrity."
      echo -e " -u \t\t Unpack the dataset."
      echo -e " -r \t\t Remove the downloaded archives."
      echo -e " -h \t\t Show this help."
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
      # Consume arguments until the next flag or end of args
      while [[ $# -gt 0 ]] && ! [[ "$1" =~ ^- ]]; do
        SUBSETS+=("$1")
        shift
      done
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

# Handle edge case: User puts download_dir at the end of the subset list
# Example: ./download.sh -s sub1 sub2 dir/
# In this case, 'dir/' is inside SUBSETS and POSITIONAL_ARGS is empty.
if [[ ${#POSITIONAL_ARGS[@]} -eq 0 ]] && [[ ${#SUBSETS[@]} -gt 0 ]]; then
    DATASET_DIR="${SUBSETS[-1]}"
    unset 'SUBSETS[${#SUBSETS[@]}-1]'
else
    DATASET_DIR="$1"
fi

if [ -z "$DATASET_DIR" ]; then
    echo "Error: No dataset directory specified."
    echo "Usage: $0 [options] <download_dir>"
    exit 1
fi

EXEC_DIR=$(pwd)

echo "MVPSP Dataset Downloader"
if [[ ${#SUBSETS[@]} -gt 0 ]]; then
  echo -e "Selected Subsets: \t${SUBSETS[*]}"
else
  echo -e "Selected Subsets: \tAll"
fi
echo -e "Download: \t\t${DOWNLOAD-false}"
echo -e "Verify: \t\t${VERIFY-false}"
echo -e "Unpack: \t\t${UNPACK-false}"
echo -e "Remove archives: \t${REMOVE_ARCHIVES-false}"
echo -e "Dataset location: \t${DATASET_DIR}"
echo ""

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
  AZCOPY_ARGS="copy --recursive --overwrite=false"

  # Handle Subsets via include-pattern
  if [[ ${#SUBSETS[@]} -gt 0 ]]; then
    # Always include the small base archive
    INCLUDE_PATTERNS="mvpsp.tar.xz"
    for subset in "${SUBSETS[@]}"; do
      INCLUDE_PATTERNS="${INCLUDE_PATTERNS};${subset}*.tar.xz"
    done
    AZCOPY_ARGS="${AZCOPY_ARGS} --include-pattern \"${INCLUDE_PATTERNS}\""
  fi

  echo "${AZCOPY} ${AZCOPY_ARGS} 'https://rocs2.blob.core.windows.net/mvpsp-public-mirror/*?sv=2022-11-02&ss=bfqt&srt=sco&sp=rlx&se=2029-07-17T17:39:08Z&st=2024-07-17T09:39:08Z&spr=https&sig=cITpUGxTo1aNHvFaMysP%2BCS%2Fy9OHfvXiwuoXGTSlBC4%3D' ."
  eval "${AZCOPY} ${AZCOPY_ARGS} 'https://rocs2.blob.core.windows.net/mvpsp-public-mirror/*?sv=2022-11-02&ss=bfqt&srt=sco&sp=rlx&se=2029-07-17T17:39:08Z&st=2024-07-17T09:39:08Z&spr=https&sig=cITpUGxTo1aNHvFaMysP%2BCS%2Fy9OHfvXiwuoXGTSlBC4%3D' ."
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
  COMMAND="echo {} ; XZ_OPT='-T0' tar -xJf {}"
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
