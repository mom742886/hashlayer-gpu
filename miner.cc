#include <cuda_runtime.h>
#include <nan.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "kernels.h"

typedef Nan::AsyncBareProgressQueueWorker<uint32_t>::ExecutionProgress MinerProgress;

class Device;

class Miner : public Nan::ObjectWrap
{
public:
  explicit Miner();
  ~Miner();

  static NAN_MODULE_INIT(Init);
  static NAN_METHOD(New);
  static NAN_METHOD(GetDevices);
  static NAN_METHOD(InitializeDevices);
  static NAN_METHOD(StartMiningOnBlock);
  static NAN_METHOD(Stop);
  bool IsMiningEnabled();
  uint64_t GetNextStartNonce(uint32_t noncesPerRun);
  uint32_t GetWorkId();

private:
  static Nan::Persistent<v8::Function> constructor;

  int numberOfDevices;
  std::vector<Device *> devices;
  bool devicesInitialized = false;

  std::atomic_bool miningEnabled;
  std::atomic_uint_fast32_t workId;
  std::atomic_uint_fast64_t startNonce;
};

class Device
{
public:
  Device(Miner *miner, uint32_t deviceIndex);
  ~Device();

  static NAN_GETTER(HandleGetters);
  static NAN_SETTER(HandleSetters);

  bool IsEnabled();
  uint32_t GetNoncesPerRun();
  uint32_t GetDeviceIndex();

  void StartMiningOnBlock(const v8::Local<v8::Function> &cbFunc, uint32_t workId, block_header *header, uint32_t nonce_offset, uint32_t difficulty_bits);
  void Initialize();
  void MineNonces(uint32_t workId, uint32_t threadIndex, block_header *header, uint32_t nonce_offset, uint32_t difficulty_bits, const MinerProgress &progress);

private:
  Miner *miner;
  uint32_t deviceIndex;
  cudaDeviceProp prop;
  bool enabled = true;
  uint32_t threads = 2;
  bool initialized = false;
  // TODO: Extract new class
  std::mutex **mutexes;
  worker_t worker;
};

class MinerWorker : public Nan::AsyncProgressQueueWorker<uint32_t>
{
public:
  MinerWorker(Nan::Callback *callback, Device *device, uint32_t workId, uint32_t threadIndex, block_header header, uint32_t nonce_offset, uint32_t difficulty_bits);

  void Execute(const MinerProgress &progress);
  void HandleProgressCallback(const uint32_t *data, size_t count);
  void HandleOKCallback();

private:
  Device *device;
  uint32_t workId;
  uint32_t threadIndex;
  block_header blockHeader;
  uint32_t nonce_offset;
  uint32_t difficulty_bits;
};

/*
* Miner
*/

Nan::Persistent<v8::Function> Miner::constructor;

Miner::Miner()
{
  cudaGetDeviceCount(&numberOfDevices);
  if (numberOfDevices < 1)
  {
    throw std::runtime_error("Could not initialize miner. No CUDA devices found.");
  }

  devices.resize(numberOfDevices);
  for (int deviceIndex = 0; deviceIndex < numberOfDevices; deviceIndex++)
  {
    devices[deviceIndex] = new Device(this, deviceIndex);
  }


  miningEnabled = false;
  workId = 0;
  startNonce = 0;
}

Miner::~Miner()
{
  for (int deviceIndex = 0; deviceIndex < numberOfDevices; deviceIndex++)
  {
    delete devices[deviceIndex];
  }
}



bool Miner::IsMiningEnabled()
{
  return miningEnabled;
}

uint64_t Miner::GetNextStartNonce(uint32_t noncesPerRun)
{
  return startNonce.fetch_add(noncesPerRun);
}

uint32_t Miner::GetWorkId()
{
  return workId;
}

NAN_MODULE_INIT(Miner::Init)
{
  v8::Local<v8::FunctionTemplate> tpl = Nan::New<v8::FunctionTemplate>(New);
  tpl->SetClassName(Nan::New("Miner").ToLocalChecked());
  tpl->InstanceTemplate()->SetInternalFieldCount(1);

  Nan::SetPrototypeMethod(tpl, "getDevices", GetDevices);
  Nan::SetPrototypeMethod(tpl, "initializeDevices", InitializeDevices);
  Nan::SetPrototypeMethod(tpl, "startMiningOnBlock", StartMiningOnBlock);
  Nan::SetPrototypeMethod(tpl, "stop", Stop);

  constructor.Reset(Nan::GetFunction(tpl).ToLocalChecked());
  Nan::Set(target, Nan::New("Miner").ToLocalChecked(), Nan::GetFunction(tpl).ToLocalChecked());
}

NAN_METHOD(Miner::New)
{
  if (!info.IsConstructCall())
  {
    return Nan::ThrowError(Nan::New("Miner() must be called with new keyword.").ToLocalChecked());
  }

  try
  {
    Miner *miner = new Miner();
    miner->Wrap(info.This());
    info.GetReturnValue().Set(info.This());
  }
  catch (std::exception &e)
  {
    return Nan::ThrowError(Nan::New(e.what()).ToLocalChecked());
  }
}

NAN_METHOD(Miner::GetDevices)
{
  Miner *miner = Nan::ObjectWrap::Unwrap<Miner>(info.This());
  v8::Local<v8::Array> devices = Nan::New<v8::Array>(miner->numberOfDevices);
  for (int deviceIndex = 0; deviceIndex < miner->numberOfDevices; deviceIndex++)
  {
    v8::Local<v8::Object> device = Nan::New<v8::Object>();
    Nan::SetPrivate(device, Nan::New("device").ToLocalChecked(), v8::External::New(info.GetIsolate(), miner->devices[deviceIndex]));
    Nan::SetAccessor(device, Nan::New("name").ToLocalChecked(), Device::HandleGetters);
    Nan::SetAccessor(device, Nan::New("clockRate").ToLocalChecked(), Device::HandleGetters);
    Nan::SetAccessor(device, Nan::New("memoryClockRate").ToLocalChecked(), Device::HandleGetters);
    Nan::SetAccessor(device, Nan::New("memoryBusWidth").ToLocalChecked(), Device::HandleGetters);
    Nan::SetAccessor(device, Nan::New("memoryBandwidth").ToLocalChecked(), Device::HandleGetters);
    Nan::SetAccessor(device, Nan::New("multiProcessorCount").ToLocalChecked(), Device::HandleGetters);
    Nan::SetAccessor(device, Nan::New("totalGlobalMem").ToLocalChecked(), Device::HandleGetters);
    Nan::SetAccessor(device, Nan::New("sharedMemPerBlock").ToLocalChecked(), Device::HandleGetters);
    Nan::SetAccessor(device, Nan::New("major").ToLocalChecked(), Device::HandleGetters);
    Nan::SetAccessor(device, Nan::New("minor").ToLocalChecked(), Device::HandleGetters);
    Nan::SetAccessor(device, Nan::New("enabled").ToLocalChecked(), Device::HandleGetters, Device::HandleSetters);
    Nan::SetAccessor(device, Nan::New("threads").ToLocalChecked(), Device::HandleGetters, Device::HandleSetters);
    Nan::Set(devices, deviceIndex, device);
  }
  info.GetReturnValue().Set(devices);
}

NAN_METHOD(Miner::InitializeDevices)
{
  Miner *miner = Nan::ObjectWrap::Unwrap<Miner>(info.This());

  if (miner->devicesInitialized)
  {
    return Nan::ThrowError(Nan::New("Devices already initialized.").ToLocalChecked());
  }

  try
  {
    for (auto device : miner->devices)
    {
      if (device->IsEnabled())
      {
        device->Initialize();
      }
    }

    miner->devicesInitialized = true;
  }
  catch (std::exception &e)
  {
    return Nan::ThrowError(Nan::New(e.what()).ToLocalChecked());
  }
}

NAN_METHOD(Miner::StartMiningOnBlock)
{
  if (!info[0]->IsUint8Array())
  {
    return Nan::ThrowError(Nan::New("Block header required.").ToLocalChecked());
  }
  v8::Local<v8::Uint8Array> blockHeader = info[0].As<v8::Uint8Array>();
  if (blockHeader->Length() > MAX_BLOCK_HEADER_SIZE)
  {
    return Nan::ThrowError(Nan::New("Block header too large.").ToLocalChecked());
  }
  if (blockHeader->Length() < 4)
  {
    return Nan::ThrowError(Nan::New("Block header too small (must have space for nonce).").ToLocalChecked());
  }

  if (!info[1]->IsUint32())
  {
    return Nan::ThrowError(Nan::New("Nonce offset required.").ToLocalChecked());
  }
  uint32_t nonce_offset = Nan::To<uint32_t>(info[1]).FromJust();

  if (!info[2]->IsUint32())
  {
    return Nan::ThrowError(Nan::New("Difficulty bits required.").ToLocalChecked());
  }
  uint32_t difficulty_bits = Nan::To<uint32_t>(info[2]).FromJust();

  if (nonce_offset + 4 > blockHeader->Length())
  {
    return Nan::ThrowError(Nan::New("Nonce offset is out of bounds.").ToLocalChecked());
  }

  if (!info[3]->IsFunction())
  {
    return Nan::ThrowError(Nan::New("Callback required.").ToLocalChecked());
  }
  v8::Local<v8::Function> cbFunc = info[3].As<v8::Function>();

  Miner *miner = Nan::ObjectWrap::Unwrap<Miner>(info.This());
  if (!miner->devicesInitialized)
  {
    return Nan::ThrowError(Nan::New("Devices are not initialized.").ToLocalChecked());
  }
  
  block_header header;
  header.data_len = blockHeader->Length();
  memcpy(header.data, blockHeader->Buffer()->GetContents().Data(), header.data_len);

  miner->miningEnabled = true;
  uint32_t workId = ++miner->workId;
  miner->startNonce = 0; // TODO: Make startNonce consistent across threads. It can be incremented by the worker mining stale block.

  int enabledDevices = 0;
  for (auto device : miner->devices)
  {
    if (device->IsEnabled())
    {
      device->StartMiningOnBlock(cbFunc, workId, &header, nonce_offset, difficulty_bits);
      enabledDevices++;
    }
  }

  if (enabledDevices == 0)
  {
    return Nan::ThrowError(Nan::New("Can't start mining - all devices are disabled.").ToLocalChecked());
  }
}

NAN_METHOD(Miner::Stop)
{
  Miner *miner = Nan::ObjectWrap::Unwrap<Miner>(info.This());
  miner->miningEnabled = false;
}

/*
* Device
*/

Device::Device(Miner *miner, uint32_t deviceIndex) : miner(miner), deviceIndex(deviceIndex)
{
  cudaGetDeviceProperties(&prop, deviceIndex);
}

Device::~Device()
{
  if (initialized)
  {
    for (uint32_t threadIndex = 0; threadIndex < threads; threadIndex++)
    {
      delete mutexes[threadIndex];
      cudaFree(worker.block_header_data[threadIndex]);
      cudaFree(worker.nonce[threadIndex]);
    }

    delete[] mutexes;
    delete[] worker.block_header_data;
    delete[] worker.nonce;
  }
}

NAN_GETTER(Device::HandleGetters)
{
  v8::Local<v8::Value> ext = Nan::GetPrivate(info.This(), Nan::New("device").ToLocalChecked()).ToLocalChecked();
  Device *device = (Device *)ext.As<v8::External>()->Value();

  std::string propertyName = std::string(*Nan::Utf8String(property));
  if (propertyName == "name")
  {
    info.GetReturnValue().Set(Nan::New(device->prop.name).ToLocalChecked());
  }
  else if (propertyName == "clockRate")
  {
    info.GetReturnValue().Set(device->prop.clockRate / 1e3); // MHz
  }
  else if (propertyName == "memoryClockRate")
  {
    info.GetReturnValue().Set(device->prop.memoryClockRate / 1e3); // MHz
  }
  else if (propertyName == "memoryBusWidth")
  {
    info.GetReturnValue().Set(device->prop.memoryBusWidth);
  }
  else if (propertyName == "memoryBandwidth")
  {
    info.GetReturnValue().Set((2.0 * device->prop.memoryClockRate * device->prop.memoryBusWidth / 8) / 1e6); // GB/s
  }
  else if (propertyName == "multiProcessorCount")
  {
    info.GetReturnValue().Set(device->prop.multiProcessorCount);
  }
  else if (propertyName == "totalGlobalMem")
  {
    info.GetReturnValue().Set((double)device->prop.totalGlobalMem);
  }
  else if (propertyName == "sharedMemPerBlock")
  {
    info.GetReturnValue().Set((double)device->prop.sharedMemPerBlock);
  }
  else if (propertyName == "major")
  {
    info.GetReturnValue().Set(device->prop.major);
  }
  else if (propertyName == "minor")
  {
    info.GetReturnValue().Set(device->prop.minor);
  }
  else if (propertyName == "enabled")
  {
    info.GetReturnValue().Set(device->enabled);
  }
  else if (propertyName == "threads")
  {
    info.GetReturnValue().Set(device->threads);
  }
}

NAN_SETTER(Device::HandleSetters)
{
  v8::Local<v8::Value> ext = Nan::GetPrivate(info.This(), Nan::New("device").ToLocalChecked()).ToLocalChecked();
  Device *device = (Device *)ext.As<v8::External>()->Value();

  std::string propertyName = std::string(*Nan::Utf8String(property));
  if (propertyName == "enabled")
  {
    if (!value->IsBoolean())
    {
      return Nan::ThrowError(Nan::New("Boolean value required.").ToLocalChecked());
    }
    device->enabled = Nan::To<bool>(value).FromJust();
  }
  else if (propertyName == "threads")
  {
    if (!value->IsUint32())
    {
      return Nan::ThrowError(Nan::New("Threads must be >= 1").ToLocalChecked());
    }
    uint32_t threads = Nan::To<uint32_t>(value).FromJust();
    if (threads < 1)
    {
      return Nan::ThrowError(Nan::New("Threads must be >= 1.").ToLocalChecked());
    }
    device->threads = threads;
  }
}

bool Device::IsEnabled()
{
  return enabled;
}

uint32_t Device::GetNoncesPerRun()
{
  return worker.nonces_per_run;
}

uint32_t Device::GetDeviceIndex()
{
  return deviceIndex;
}

void Device::StartMiningOnBlock(const v8::Local<v8::Function> &cbFunc, uint32_t workId, block_header *header, uint32_t nonce_offset, uint32_t difficulty_bits)
{
  Nan::HandleScope scope;

  for (uint32_t threadIndex = 0; threadIndex < threads; threadIndex++)
  {
    Nan::AsyncQueueWorker(new MinerWorker(new Nan::Callback(cbFunc), this, workId, threadIndex, *header, nonce_offset, difficulty_bits));
  }
}

void Device::Initialize()
{
  if (initialized)
  {
    return;
  }

  cudaSetDevice(deviceIndex);
  cudaDeviceReset();
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

  // Configure kernel launch parameters for Blake2b mining
  // Each thread processes one nonce
  uint32_t threadsPerBlock = 256; // Optimal for most GPUs
  uint32_t blocksPerGrid = prop.multiProcessorCount * 4; // Adjust based on GPU
  uint32_t noncesPerRun = blocksPerGrid * threadsPerBlock;

  worker.nonces_per_run = noncesPerRun;
  worker.mine_blocks = dim3(blocksPerGrid);
  worker.mine_threads = dim3(threadsPerBlock);

  mutexes = new std::mutex *[threads];
  worker.block_header_data = new uint64_t *[threads];
  worker.nonce = new uint32_t *[threads];

  // Allocate GPU memory for block header (max size)
  uint32_t header_qwords = (MAX_BLOCK_HEADER_SIZE + 7) / 8;

  for (uint32_t threadIndex = 0; threadIndex < threads; threadIndex++)
  {
    mutexes[threadIndex] = new std::mutex();

    cudaError_t result = cudaMalloc(&worker.block_header_data[threadIndex], header_qwords * sizeof(uint64_t));
    if (result != cudaSuccess)
    {
      throw std::runtime_error("Could not allocate GPU memory for block header.");
    }

    result = cudaMalloc(&worker.nonce[threadIndex], sizeof(uint32_t));
    if (result != cudaSuccess)
    {
      throw std::runtime_error("Could not allocate GPU memory for nonce.");
    }
  }

  initialized = true;
}

void Device::MineNonces(uint32_t workId, uint32_t threadIndex, block_header *header, uint32_t nonce_offset, uint32_t difficulty_bits, const MinerProgress &progress)
{
  std::lock_guard<std::mutex> lock(*mutexes[threadIndex]);

  cudaSetDevice(deviceIndex);

  // Set block header and mining parameters
  set_block_header(&worker, threadIndex, header);
  worker.nonce_offset = nonce_offset;
  worker.difficulty_bits = difficulty_bits;

  while (miner->IsMiningEnabled())
  {
    if (workId != miner->GetWorkId())
    {
      break;
    }
    uint32_t noncesPerRun = GetNoncesPerRun();
    uint64_t startNonce = miner->GetNextStartNonce(noncesPerRun);
    if (startNonce + noncesPerRun > UINT32_MAX)
    {
      break;
    }

    uint32_t nonce;
    cudaError_t result = mine_nonces(&worker, threadIndex, (uint32_t)startNonce, &nonce);
    if (result != cudaSuccess)
    {
      const char *errorMsg = cudaGetErrorString(result);
      std::cerr << "GPU #" << deviceIndex << " failed: " << errorMsg << "\n";
      std::exit(result);
    }

    progress.Send(&nonce, 1);
  }
}

/*
* MinerWorker
*/

MinerWorker::MinerWorker(Nan::Callback *callback, Device *device, uint32_t workId, uint32_t threadIndex, block_header header, uint32_t nonce_offset, uint32_t difficulty_bits)
    : AsyncProgressQueueWorker(callback), device(device), workId(workId), threadIndex(threadIndex), blockHeader(header), nonce_offset(nonce_offset), difficulty_bits(difficulty_bits)
{
}

void MinerWorker::Execute(const MinerProgress &progress)
{
  try
  {
    device->MineNonces(workId, threadIndex, &blockHeader, nonce_offset, difficulty_bits, progress);
  }
  catch (std::exception &e)
  {
    SetErrorMessage(e.what());
  }
}

void MinerWorker::HandleProgressCallback(const uint32_t *nonce, size_t count)
{
  Nan::HandleScope scope;

  v8::Local<v8::Object> obj = Nan::New<v8::Object>();
  Nan::Set(obj, Nan::New("done").ToLocalChecked(), Nan::New(false));
  Nan::Set(obj, Nan::New("device").ToLocalChecked(), Nan::New(device->GetDeviceIndex()));
  Nan::Set(obj, Nan::New("thread").ToLocalChecked(), Nan::New(threadIndex));
  Nan::Set(obj, Nan::New("noncesPerRun").ToLocalChecked(), Nan::New(device->GetNoncesPerRun()));
  Nan::Set(obj, Nan::New("nonce").ToLocalChecked(), Nan::New(*nonce));

  v8::Local<v8::Value> argv[] = {Nan::Null(), obj};
  callback->Call(2, argv, async_resource);
}

void MinerWorker::HandleOKCallback()
{
  Nan::HandleScope scope;

  v8::Local<v8::Object> obj = Nan::New<v8::Object>();
  Nan::Set(obj, Nan::New("done").ToLocalChecked(), Nan::New(true));
  Nan::Set(obj, Nan::New("device").ToLocalChecked(), Nan::New(device->GetDeviceIndex()));
  Nan::Set(obj, Nan::New("thread").ToLocalChecked(), Nan::New(threadIndex));
  Nan::Set(obj, Nan::New("noncesPerRun").ToLocalChecked(), Nan::New(device->GetNoncesPerRun()));
  Nan::Set(obj, Nan::New("nonce").ToLocalChecked(), Nan::Undefined());

  v8::Local<v8::Value> argv[] = {Nan::Null(), obj};
  callback->Call(2, argv, async_resource);
}

NODE_MODULE(blake2b_miner_cuda, Miner::Init);