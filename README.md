# Blake2b GPU Miner

GPU miner for Blake2b algorithm using CUDA. This project provides a high-performance GPU mining solution for Blake2b hashing.

## Requirements

- Node.js >= 12.0.0
- CUDA Toolkit (with nvcc compiler)
- node-gyp
- npm or yarn

## Installation

1. Install dependencies:
```bash
npm install
```

This will automatically:
- Compile CUDA files (.cu) to object files
- Build the Node.js native addon

2. Build manually (if needed):
```bash
npm run build
```

## Usage

### Basic Example

```javascript
const NativeMiner = require('./native.js');

const deviceOptions = {
    forDevice: (index) => ({
        enabled: true,
        threads: 2
    })
};

const miner = new NativeMiner('cuda', deviceOptions);

// Set up hash rate reporting
miner.onHashrateChanged = (hashRates) => {
    hashRates.forEach((rate, idx) => {
        if (rate) {
            console.log(`GPU #${idx}: ${(rate / 1000000).toFixed(2)} MH/s`);
        }
    });
};

// Create block header (with nonce = 0)
const blockHeader = new Uint8Array(80);
// ... populate block header with your data ...

// Set nonce to 0 at offset 41
const nonceView = new DataView(blockHeader.buffer, blockHeader.byteOffset + 41, 4);
nonceView.setUint32(0, 0, true);

// Start mining
miner.startMiningOnBlock(
    blockHeader,    // Block header bytes
    41,             // Nonce offset (byte position)
    20,             // Difficulty bits (number of leading zero bits required)
    (result) => {
        if (result.nonce > 0) {
            console.log(`Solution found! Nonce: ${result.nonce}`);
            // Update block header with found nonce
            nonceView.setUint32(0, result.nonce, true);
        }
    }
);

// Stop mining when done
// miner.stop();
```

### Run Example App

```bash
npm test
# or
node app.js
```

## API

### `NativeMiner`

#### Constructor
```javascript
new NativeMiner(type, deviceOptions)
```
- `type`: String - 'cuda' for CUDA devices
- `deviceOptions`: Object - Configuration for each GPU device

#### Methods

##### `startMiningOnBlock(blockHeader, nonce_offset, difficulty_bits, callback)`
Start mining on a block header.

- `blockHeader`: `Uint8Array` - Block header bytes (with nonce initially set to 0)
- `nonce_offset`: `number` - Byte offset where nonce is located in block header
- `difficulty_bits`: `number` - Number of leading zero bits required in hash
- `callback`: `Function` - Callback function that receives mining results

Callback receives an object with:
- `done`: `boolean` - Whether mining is complete
- `device`: `number` - GPU device index
- `thread`: `number` - Thread index
- `nonce`: `number` - Found nonce (0 if not found)
- `noncesPerRun`: `number` - Number of nonces checked in this run

##### `stop()`
Stop mining.

#### Properties

##### `devices`
Array of GPU device objects with properties:
- `name`: Device name
- `multiProcessorCount`: Number of multiprocessors
- `clockRate`: Clock rate in MHz
- `totalGlobalMem`: Total global memory in bytes
- `enabled`: Whether device is enabled
- `threads`: Number of mining threads

##### `onHashrateChanged`
Callback function that receives hash rates array (one per device).

## Building

### Manual Build

```bash
# Clean previous build
npm run clean

# Rebuild
npm run rebuild
```

### Troubleshooting

#### CUDA not found
Make sure CUDA Toolkit is installed and `nvcc` is in your PATH, or set `CUDA_PATH` environment variable.

#### Compilation errors
- Ensure you have the correct CUDA version for your GPU
- Check that all source files are present
- Verify node-gyp is properly installed: `npm install -g node-gyp`

## Project Structure

```
.
├── blake2b.cu          # Blake2b hash implementation (CUDA)
├── kernels.cu         # CUDA kernel wrappers
├── kernels.h           # Kernel definitions and structures
├── miner.cc            # Node.js native addon implementation
├── native.js           # JavaScript wrapper
├── app.js              # Example application
├── binding.gyp         # Build configuration
└── package.json        # Project configuration
```

## License

MIT
