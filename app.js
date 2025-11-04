const NativeMiner = require('./native.js');

// Example device configuration
const deviceOptions = {
    forDevice: (index) => {
        // Configure each device individually
        return {
            enabled: true,      // Enable this device
            threads: 2          // Number of mining threads per device
        };
    }
};

// Create miner instance
const miner = new NativeMiner('cuda', deviceOptions);

// Print device information
console.log('\n=== Available GPU Devices ===');
miner.devices.forEach((device, idx) => {
    if (device.enabled) {
        console.log(`GPU #${idx}: ${device.name}`);
        console.log(`  - Multiprocessors: ${device.multiProcessorCount}`);
        console.log(`  - Clock Rate: ${device.clockRate} MHz`);
        console.log(`  - Memory: ${(device.totalGlobalMem / 1024 / 1024 / 1024).toFixed(2)} GB`);
        console.log(`  - Threads: ${device.threads}`);
    }
});
console.log('');

// Set up hash rate reporting
miner.onHashrateChanged = (hashRates) => {
    hashRates.forEach((rate, idx) => {
        if (rate) {
            const hashesPerSecond = rate.toFixed(2);
            const kHashes = (rate / 1000).toFixed(2);
            const mHashes = (rate / 1000000).toFixed(2);
            let display = `${hashesPerSecond} H/s`;
            if (rate >= 1000000) {
                display = `${mHashes} MH/s`;
            } else if (rate >= 1000) {
                display = `${kHashes} kH/s`;
            }
            console.log(`GPU #${idx}: ${display}`);
        }
    });
};

// Example: Create a block header
// In a real application, this would come from your blockchain
function createBlockHeader() {
    // Example block header: 80 bytes total
    // You should replace this with actual block header data from your blockchain
    const header = new Uint8Array(80);
    
    // Fill with some example data (all zeros for demo)
    // In practice, you would populate this with:
    // - Previous block hash
    // - Merkle root
    // - Timestamp
    // - Difficulty
    // - Nonce (initially set to 0)
    
    // For demo purposes, set some dummy values
    for (let i = 0; i < header.length; i++) {
        header[i] = i % 256; // Dummy data
    }
    
    return header;
}

// Start mining
function startMining() {
    console.log('=== Starting Mining ===\n');
    
    const blockHeader = createBlockHeader();
    const nonce_offset = 41; // Nonce is typically at bytes 76-79 in a standard block header
    const difficulty_bits = 20; // Require 20 leading zero bits (adjust based on your needs)
    
    // Ensure nonce is set to 0 initially
    // Write nonce as little-endian uint32 at nonce_offset
    const nonceView = new DataView(blockHeader.buffer, blockHeader.byteOffset + nonce_offset, 4);
    nonceView.setUint32(0, 0, true); // true = little-endian
    
    console.log(`Block header size: ${blockHeader.length} bytes`);
    console.log(`Nonce offset: ${nonce_offset}`);
    console.log(`Difficulty: ${difficulty_bits} leading zero bits`);
    console.log('Starting to mine...\n');
    
    // Start mining with callback
    miner.startMiningOnBlock(blockHeader, nonce_offset, difficulty_bits, (result) => {
        if (result.nonce > 0) {
            console.log('\n=== SOLUTION FOUND! ===');
            console.log(`Device: GPU #${result.device}`);
            console.log(`Thread: ${result.thread}`);
            console.log(`Nonce: ${result.nonce}`);
            console.log(`Nonces checked: ${result.noncesPerRun}`);
            console.log('');
            
            // Update the block header with the found nonce
            nonceView.setUint32(0, result.nonce, true);
            
            // Here you would typically:
            // 1. Submit the solution to your blockchain network
            // 2. Start mining on the next block
            
            // For demo, we'll continue mining
            // miner.stop(); // Uncomment to stop after finding a solution
        }
    });
}

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('\n\nStopping miner...');
    miner.stop();
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\n\nStopping miner...');
    miner.stop();
    process.exit(0);
});

// Start the mining process
if (miner.devices.some(d => d.enabled)) {
    startMining();
} else {
    console.error('No enabled GPU devices found!');
    process.exit(1);
}
