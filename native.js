class NativeMiner {

    /**
     * @param {string} type 
     * @param {object} deviceOptions 
     */
    constructor(type, deviceOptions) {
        const Miner = require('bindings')(`blake2b_miner_${type}.node`);
        this._nativeMiner = new Miner.Miner();

        this._devices = this._nativeMiner.getDevices();
        this._devices.forEach((device, idx) => {
            const options = deviceOptions ? deviceOptions.forDevice ? deviceOptions.forDevice(idx) : (deviceOptions[idx] || {}) : {};
            if (!options.enabled) {
                device.enabled = false;
                console.log(`GPU #${idx}: ${device.name}. Disabled by user.`);
                return;
            }
            if (options.threads !== undefined) {
                device.threads = options.threads;
            }
            if (type === 'cuda') {
                console.log(`GPU #${idx}: ${device.name}, ${device.multiProcessorCount} SM @ ${device.clockRate} MHz. (threads: ${device.threads})`);
            } else {
                console.log(`GPU #${idx}: ${device.name}, ${device.maxComputeUnits} CU @ ${device.maxClockFrequency} MHz. (threads: ${device.threads})`);
            }
        });

        const threads = this._devices.reduce((threads, device) => threads + (device.enabled ? device.threads : 0), 4); // 4 initial threads + more for GPU workers
        process.env.UV_THREADPOOL_SIZE = threads;
        this._nativeMiner.initializeDevices();
        this._hashes = [];
        this._lastHashRates = [];
    }

    _reportHashRate() {
        const averageHashRates = [];
        this._hashes.forEach((hashes, idx) => {
            const hashRate = hashes / NativeMiner.HASHRATE_REPORT_INTERVAL;
            this._lastHashRates[idx] = this._lastHashRates[idx] || [];
            this._lastHashRates[idx].push(hashRate);
            if (this._lastHashRates[idx].length > NativeMiner.HASHRATE_MOVING_AVERAGE) {
                this._lastHashRates[idx].shift();
                averageHashRates[idx] = this._lastHashRates[idx].reduce((sum, val) => sum + val, 0) / this._lastHashRates[idx].length;
            } else if (this._lastHashRates[idx].length > 1) {
                averageHashRates[idx] = this._lastHashRates[idx].slice(1).reduce((sum, val) => sum + val, 0) / (this._lastHashRates[idx].length - 1);
            }
        });
        this._hashes = [];
        if (averageHashRates.length > 0) {
            if (this.onHashrateChanged) {
                this.onHashrateChanged(averageHashRates);
            }
        }
    }

    /**
     * Start mining on a block
     * @param {Uint8Array} blockHeader - Block header bytes (with nonce set to 0)
     * @param {number} nonce_offset - Byte offset where nonce is located in block header
     * @param {number} difficulty_bits - Number of leading zero bits required
     * @param {Function} callback - Callback function that receives mining results
     */
    startMiningOnBlock(blockHeader, nonce_offset, difficulty_bits, callback) {
        if (!this._hashRateTimer) {
            this._hashRateTimer = setInterval(() => this._reportHashRate(), 1000 * NativeMiner.HASHRATE_REPORT_INTERVAL);
        }
        this._nativeMiner.startMiningOnBlock(blockHeader, nonce_offset, difficulty_bits, (error, obj) => {
            if (error) {
                throw error;
            }
            if (obj.done === true) {
                return;
            }
            this._hashes[obj.device] = (this._hashes[obj.device] || 0) + obj.noncesPerRun;
            if (callback) {
                callback(obj);
            }
        });
    }

    stop() {
        this._nativeMiner.stop();
        if (this._hashRateTimer) {
            this._hashes = [];
            this._lastHashRates = [];
            clearInterval(this._hashRateTimer);
            delete this._hashRateTimer;
        }
    }

    get devices() {
        return this._devices;
    }
}

NativeMiner.HASHRATE_MOVING_AVERAGE = 6; // measurements
NativeMiner.HASHRATE_REPORT_INTERVAL = 10; // seconds

module.exports = NativeMiner;