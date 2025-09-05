#!/usr/bin/env node
/**
 * LuminaAI Desktop Startup Script
 * Handles Python dependency installation and backend startup
 */

const { spawn, exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

class LuminaAIStarter {
    constructor() {
        this.pythonCmd = null;
        this.backendProcess = null;
    }

    async findPython() {
        const commands = ['python3', 'python', 'py'];
        
        for (const cmd of commands) {
            try {
                const result = await this.execCommand(`${cmd} --version`);
                if (result.includes('Python')) {
                    this.pythonCmd = cmd;
                    console.log(`✅ Found Python: ${result.trim()}`);
                    return true;
                }
            } catch (error) {
                // Command not found, try next
                continue;
            }
        }
        
        console.error('❌ Python not found. Please install Python 3.7+ and try again.');
        return false;
    }

    async checkPythonPackages() {
        const requiredPackages = [
            'torch',
            'numpy', 
            'flask',
            'flask-socketio',
            'flask-cors'
        ];

        console.log('🔍 Checking Python packages...');
        
        const missingPackages = [];
        
        for (const pkg of requiredPackages) {
            try {
                await this.execCommand(`${this.pythonCmd} -c "import ${pkg.replace('-', '_')}"`);
                console.log(`✅ ${pkg} - installed`);
            } catch (error) {
                console.log(`❌ ${pkg} - missing`);
                missingPackages.push(pkg);
            }
        }

        return missingPackages;
    }

    async installPackages(packages) {
        if (packages.length === 0) return true;

        console.log(`📦 Installing missing packages: ${packages.join(', ')}`);
        
        const installCmd = `${this.pythonCmd} -m pip install ${packages.join(' ')}`;
        
        try {
            await this.execCommand(installCmd, { timeout: 300000 }); // 5 minute timeout
            console.log('✅ Packages installed successfully');
            return true;
        } catch (error) {
            console.error('❌ Failed to install packages:', error.message);
            console.error('Please install manually:');
            console.error(`  ${installCmd}`);
            return false;
        }
    }

    async startBackend() {
        return new Promise((resolve, reject) => {
            console.log('🚀 Starting Python backend...');
            
            const chatScriptPath = path.join(__dirname, 'chat_server.py');
            
            if (!fs.existsSync(chatScriptPath)) {
                reject(new Error('chat_server.py not found'));
                return;
            }

            this.backendProcess = spawn(this.pythonCmd, [
                chatScriptPath,
                '--server-mode',
                '--host', 'localhost',
                '--port', '5001'
            ], {
                cwd: __dirname,
                stdio: ['pipe', 'pipe', 'pipe']
            });

            let serverReady = false;

            this.backendProcess.stdout.on('data', (data) => {
                const output = data.toString();
                console.log('[Backend]:', output.trim());
                
                if (output.includes('Backend ready') || output.includes('Running on')) {
                    if (!serverReady) {
                        serverReady = true;
                        resolve();
                    }
                }
            });

            this.backendProcess.stderr.on('data', (data) => {
                const error = data.toString();
                console.error('[Backend Error]:', error.trim());
            });

            this.backendProcess.on('close', (code) => {
                console.log(`Backend exited with code ${code}`);
                if (!serverReady) {
                    reject(new Error(`Backend exited with code ${code}`));
                }
            });

            this.backendProcess.on('error', (error) => {
                console.error('Failed to start backend:', error);
                reject(error);
            });

            // Timeout after 30 seconds
            setTimeout(() => {
                if (!serverReady) {
                    console.log('⚠️  Backend startup timeout - assuming ready');
                    resolve();
                }
            }, 30000);
        });
    }

    async startElectron() {
        console.log('⚡ Starting Electron app...');
        
        const electronCmd = process.platform === 'win32' ? 'electron.cmd' : 'electron';
        const electronPath = path.join(__dirname, 'node_modules', '.bin', electronCmd);
        
        const electronProcess = spawn(electronPath, ['.'], {
            cwd: __dirname,
            stdio: 'inherit'
        });

        electronProcess.on('close', (code) => {
            console.log('Electron app closed');
            this.cleanup();
        });

        electronProcess.on('error', (error) => {
            console.error('Failed to start Electron:', error);
            this.cleanup();
        });
    }

    cleanup() {
        if (this.backendProcess && !this.backendProcess.killed) {
            console.log('🧹 Cleaning up backend process...');
            this.backendProcess.kill('SIGTERM');
            
            setTimeout(() => {
                if (!this.backendProcess.killed) {
                    this.backendProcess.kill('SIGKILL');
                }
            }, 5000);
        }
    }

    execCommand(command, options = {}) {
        return new Promise((resolve, reject) => {
            exec(command, options, (error, stdout, stderr) => {
                if (error) {
                    reject(error);
                } else {
                    resolve(stdout || stderr);
                }
            });
        });
    }

    async run() {
        console.log('🌟 LuminaAI Desktop - Starting...');
        console.log('================================');

        try {
            // Check Python
            const pythonFound = await this.findPython();
            if (!pythonFound) {
                process.exit(1);
            }

            // Check packages
            const missingPackages = await this.checkPythonPackages();
            
            if (missingPackages.length > 0) {
                const shouldInstall = await this.promptUser(
                    `Missing packages detected: ${missingPackages.join(', ')}\nInstall automatically? (y/n): `
                );
                
                if (shouldInstall.toLowerCase().startsWith('y')) {
                    const installed = await this.installPackages(missingPackages);
                    if (!installed) {
                        process.exit(1);
                    }
                } else {
                    console.log('Please install the required packages manually and try again.');
                    process.exit(1);
                }
            }

            // Start backend
            await this.startBackend();
            
            // Small delay to ensure backend is ready
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // Start Electron
            await this.startElectron();

        } catch (error) {
            console.error('❌ Startup failed:', error.message);
            this.cleanup();
            process.exit(1);
        }
    }

    promptUser(question) {
        return new Promise((resolve) => {
            const readline = require('readline');
            const rl = readline.createInterface({
                input: process.stdin,
                output: process.stdout
            });

            rl.question(question, (answer) => {
                rl.close();
                resolve(answer);
            });
        });
    }
}

// Handle process termination
process.on('SIGINT', () => {
    console.log('\n👋 Shutting down LuminaAI Desktop...');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\n👋 Shutting down LuminaAI Desktop...');
    process.exit(0);
});

// Run the starter
const starter = new LuminaAIStarter();
starter.run().catch((error) => {
    console.error('Fatal error:', error);
    process.exit(1);
});