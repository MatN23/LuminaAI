const { app, BrowserWindow, Menu, dialog, ipcMain, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');

class LuminaAIApp {
    constructor() {
        this.mainWindow = null;
        this.pythonProcess = null;
        this.isQuitting = false;
    }

    async createWindow() {
        // Create the browser window
        this.mainWindow = new BrowserWindow({
            width: 1400,
            height: 900,
            minWidth: 1000,
            minHeight: 700,
            webPreferences: {
                nodeIntegration: true,
                contextIsolation: false,
                enableRemoteModule: true
            },
            titleBarStyle: 'hiddenInset',
            frame: process.platform !== 'darwin',
            show: false,
            backgroundColor: '#0c0c0f',
            icon: path.join(__dirname, 'assets', 'icon.png') // You can add an icon file
        });

        // Load the HTML file
        await this.mainWindow.loadFile('index.html');

        // Show window when ready to prevent visual flash
        this.mainWindow.once('ready-to-show', () => {
            this.mainWindow.show();
            
            // Focus on macOS
            if (process.platform === 'darwin') {
                app.focus();
            }
        });

        // Handle window closed
        this.mainWindow.on('closed', () => {
            this.mainWindow = null;
        });

        // Handle window close attempt
        this.mainWindow.on('close', (event) => {
            if (process.platform === 'darwin' && !this.isQuitting) {
                event.preventDefault();
                this.mainWindow.hide();
            }
        });

        // Open external links in default browser
        this.mainWindow.webContents.setWindowOpenHandler(({ url }) => {
            shell.openExternal(url);
            return { action: 'deny' };
        });

        // Development tools in development mode
        if (process.env.NODE_ENV === 'development') {
            this.mainWindow.webContents.openDevTools();
        }
    }

    createMenu() {
        const template = [
            {
                label: 'File',
                submenu: [
                    {
                        label: 'Load Model...',
                        accelerator: 'CmdOrCtrl+O',
                        click: async () => {
                            await this.loadModelDialog();
                        }
                    },
                    {
                        label: 'Load Checkpoint...',
                        accelerator: 'CmdOrCtrl+Shift+O',
                        click: async () => {
                            await this.loadCheckpointDialog();
                        }
                    },
                    { type: 'separator' },
                    {
                        label: 'Export Chat...',
                        accelerator: 'CmdOrCtrl+E',
                        click: () => {
                            // Send to renderer
                            this.mainWindow.webContents.send('export-chat');
                        }
                    },
                    { type: 'separator' },
                    {
                        label: 'Quit',
                        accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
                        click: () => {
                            this.isQuitting = true;
                            app.quit();
                        }
                    }
                ]
            },
            {
                label: 'Edit',
                submenu: [
                    { role: 'undo' },
                    { role: 'redo' },
                    { type: 'separator' },
                    { role: 'cut' },
                    { role: 'copy' },
                    { role: 'paste' },
                    { role: 'selectall' }
                ]
            },
            {
                label: 'View',
                submenu: [
                    { role: 'reload' },
                    { role: 'forceReload' },
                    { role: 'toggleDevTools' },
                    { type: 'separator' },
                    { role: 'resetZoom' },
                    { role: 'zoomIn' },
                    { role: 'zoomOut' },
                    { type: 'separator' },
                    { role: 'togglefullscreen' }
                ]
            },
            {
                label: 'Chat',
                submenu: [
                    {
                        label: 'Clear History',
                        accelerator: 'CmdOrCtrl+K',
                        click: () => {
                            this.mainWindow.webContents.send('clear-chat');
                        }
                    },
                    {
                        label: 'New Chat',
                        accelerator: 'CmdOrCtrl+N',
                        click: () => {
                            this.mainWindow.webContents.send('new-chat');
                        }
                    },
                    { type: 'separator' },
                    {
                        label: 'Model Info',
                        accelerator: 'CmdOrCtrl+I',
                        click: () => {
                            this.mainWindow.webContents.send('show-model-info');
                        }
                    }
                ]
            },
            {
                label: 'Window',
                submenu: [
                    { role: 'minimize' },
                    { role: 'close' }
                ]
            },
            {
                role: 'help',
                submenu: [
                    {
                        label: 'About LuminaAI',
                        click: () => {
                            this.showAbout();
                        }
                    },
                    {
                        label: 'GitHub Repository',
                        click: async () => {
                            await shell.openExternal('https://github.com/your-repo/luminaai');
                        }
                    }
                ]
            }
        ];

        // macOS specific menu adjustments
        if (process.platform === 'darwin') {
            template.unshift({
                label: app.getName(),
                submenu: [
                    { role: 'about' },
                    { type: 'separator' },
                    { role: 'services' },
                    { type: 'separator' },
                    { role: 'hide' },
                    { role: 'hideOthers' },
                    { role: 'unhide' },
                    { type: 'separator' },
                    { role: 'quit' }
                ]
            });

            // Window menu
            template[5].submenu = [
                { role: 'close' },
                { role: 'minimize' },
                { role: 'zoom' },
                { type: 'separator' },
                { role: 'front' }
            ];
        }

        const menu = Menu.buildFromTemplate(template);
        Menu.setApplicationMenu(menu);
    }

    async loadModelDialog() {
        const result = await dialog.showOpenDialog(this.mainWindow, {
            title: 'Select Model File',
            buttonLabel: 'Load Model',
            filters: [
                { name: 'PyTorch Models', extensions: ['pt', 'pth', 'bin', 'safetensors'] },
                { name: 'All Files', extensions: ['*'] }
            ],
            properties: ['openFile']
        });

        if (!result.canceled && result.filePaths.length > 0) {
            const modelPath = result.filePaths[0];
            
            // Check if file exists
            if (fs.existsSync(modelPath)) {
                // Send to renderer process
                this.mainWindow.webContents.send('load-model', modelPath);
            } else {
                dialog.showErrorBox('File Not Found', `The selected model file does not exist: ${modelPath}`);
            }
        }
    }

    async loadCheckpointDialog() {
        const result = await dialog.showOpenDialog(this.mainWindow, {
            title: 'Select Checkpoint Directory',
            buttonLabel: 'Load Checkpoint',
            properties: ['openDirectory']
        });

        if (!result.canceled && result.filePaths.length > 0) {
            const checkpointDir = result.filePaths[0];
            
            // Look for checkpoint files in the directory
            try {
                const files = fs.readdirSync(checkpointDir);
                const checkpointFiles = files.filter(file => 
                    file.endsWith('.pt') || file.endsWith('.pth') || file.includes('checkpoint')
                );

                if (checkpointFiles.length > 0) {
                    // Find the best checkpoint
                    let bestCheckpoint = checkpointFiles.find(f => f.includes('best'));
                    if (!bestCheckpoint) {
                        // Sort by modification time, newest first
                        bestCheckpoint = checkpointFiles
                            .map(f => ({ name: f, path: path.join(checkpointDir, f) }))
                            .sort((a, b) => fs.statSync(b.path).mtime - fs.statSync(a.path).mtime)[0].name;
                    }

                    const checkpointPath = path.join(checkpointDir, bestCheckpoint);
                    this.mainWindow.webContents.send('load-model', checkpointPath);
                } else {
                    dialog.showErrorBox('No Checkpoints Found', 'No checkpoint files found in the selected directory.');
                }
            } catch (error) {
                dialog.showErrorBox('Error', `Failed to read checkpoint directory: ${error.message}`);
            }
        }
    }

    showAbout() {
        dialog.showMessageBox(this.mainWindow, {
            type: 'info',
            title: 'About LuminaAI',
            message: 'LuminaAI Neural Desktop Interface',
            detail: `Version: 1.0.0
Built with Electron and PyTorch
Advanced neural conversation interface with transformer models

© 2024 Your Name`,
            buttons: ['OK']
        });
    }

    async startPythonBackend() {
        return new Promise((resolve, reject) => {
            // Check if Chat.py exists
            const chatScriptPath = path.join(__dirname, 'Chat.py');
            if (!fs.existsSync(chatScriptPath)) {
                console.error('Chat.py not found in application directory');
                reject(new Error('Chat.py not found'));
                return;
            }

            console.log('Starting Python backend...');
            
            // Try different Python commands
            const pythonCommands = ['python3', 'python', 'py'];
            let pythonCmd = null;

            // Find available Python command
            for (const cmd of pythonCommands) {
                try {
                    const testProcess = spawn(cmd, ['--version'], { stdio: 'pipe' });
                    testProcess.on('close', (code) => {
                        if (code === 0 && !pythonCmd) {
                            pythonCmd = cmd;
                        }
                    });
                    testProcess.on('error', () => {
                        // Command not available
                    });
                } catch (e) {
                    // Command not available
                }
            }

            // Use python3 as fallback
            if (!pythonCmd) pythonCmd = 'python3';

            // Start the server mode of Chat.py
            this.pythonProcess = spawn(pythonCmd, [
                chatScriptPath,
                '--server-mode',
                '--port', '5001'
            ], {
                cwd: __dirname,
                stdio: ['pipe', 'pipe', 'pipe']
            });

            this.pythonProcess.stdout.on('data', (data) => {
                const output = data.toString();
                console.log('[Python Backend]:', output);
                
                // Look for server ready message
                if (output.includes('Server running') || output.includes('Backend ready')) {
                    resolve();
                }
            });

            this.pythonProcess.stderr.on('data', (data) => {
                const error = data.toString();
                console.error('[Python Backend Error]:', error);
                
                // Don't reject on warnings
                if (!error.toLowerCase().includes('warning')) {
                    reject(new Error(error));
                }
            });

            this.pythonProcess.on('close', (code) => {
                console.log(`Python backend exited with code ${code}`);
                if (code !== 0 && !this.isQuitting) {
                    reject(new Error(`Python backend exited with code ${code}`));
                }
            });

            this.pythonProcess.on('error', (error) => {
                console.error('Failed to start Python backend:', error);
                reject(error);
            });

            // Timeout after 10 seconds
            setTimeout(() => {
                if (this.pythonProcess && this.pythonProcess.exitCode === null) {
                    console.log('Python backend started (timeout reached, assuming success)');
                    resolve();
                }
            }, 10000);
        });
    }

    setupIPCHandlers() {
        // Handle requests from renderer process
        ipcMain.handle('get-app-version', () => {
            return app.getVersion();
        });

        ipcMain.handle('get-user-data-path', () => {
            return app.getPath('userData');
        });

        ipcMain.handle('show-error-dialog', (event, title, content) => {
            dialog.showErrorBox(title, content);
        });

        ipcMain.handle('show-save-dialog', async (event, options) => {
            const result = await dialog.showSaveDialog(this.mainWindow, options);
            return result;
        });

        ipcMain.handle('write-file', (event, filePath, data) => {
            try {
                fs.writeFileSync(filePath, data, 'utf8');
                return { success: true };
            } catch (error) {
                return { success: false, error: error.message };
            }
        });
    }

    async initialize() {
        // Set up IPC handlers
        this.setupIPCHandlers();

        // Start Python backend
        try {
            await this.startPythonBackend();
            console.log('Python backend started successfully');
        } catch (error) {
            console.error('Failed to start Python backend:', error);
            
            // Show error dialog
            const choice = await dialog.showMessageBox(null, {
                type: 'error',
                title: 'Backend Error',
                message: 'Failed to start Python backend',
                detail: `Error: ${error.message}

Make sure Python and required packages are installed:
- torch
- numpy
- flask
- flask-socketio

Would you like to continue without the backend? (Limited functionality)`,
                buttons: ['Quit', 'Continue'],
                defaultId: 0,
                cancelId: 0
            });

            if (choice.response === 0) {
                app.quit();
                return;
            }
        }

        // Create window
        await this.createWindow();

        // Create menu
        this.createMenu();
    }

    cleanup() {
        if (this.pythonProcess && !this.pythonProcess.killed) {
            console.log('Terminating Python backend...');
            this.pythonProcess.kill('SIGTERM');
            
            // Force kill after 5 seconds
            setTimeout(() => {
                if (!this.pythonProcess.killed) {
                    this.pythonProcess.kill('SIGKILL');
                }
            }, 5000);
        }
    }
}

// App initialization
const luminaApp = new LuminaAIApp();

// App event handlers
app.whenReady().then(async () => {
    await luminaApp.initialize();
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        luminaApp.cleanup();
        app.quit();
    }
});

app.on('activate', async () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        await luminaApp.createWindow();
    } else if (luminaApp.mainWindow) {
        luminaApp.mainWindow.show();
    }
});

app.on('before-quit', () => {
    luminaApp.isQuitting = true;
    luminaApp.cleanup();
});

// Handle certificate errors for development
app.on('certificate-error', (event, webContents, url, error, certificate, callback) => {
    if (process.env.NODE_ENV === 'development') {
        event.preventDefault();
        callback(true);
    } else {
        callback(false);
    }
});

// Security: prevent new window creation
app.on('web-contents-created', (event, contents) => {
    contents.on('new-window', (event, navigationUrl) => {
        event.preventDefault();
        shell.openExternal(navigationUrl);
    });
});