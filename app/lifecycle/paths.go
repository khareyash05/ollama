package lifecycle

import (
    "errors"
    "fmt"
    "log/slog"
    "os"
    "path/filepath"
    "runtime"
    "strings"
)

var (
    AppName    = "ollama app"
    CLIName    = "ollama"
    AppDir     = "/opt/Ollama"
    AppDataDir = "/opt/Ollama"
    // TODO - should there be a distinct log dir?
    UpdateStageDir   = "/tmp"
    AppLogFile       = "/tmp/ollama_app.log"
    ServerLogFile    = "/tmp/ollama.log"
    UpgradeLogFile   = "/tmp/ollama_update.log"
    Installer        = "OllamaSetup.exe"
    LogRotationCount = 5

    getEnv        = os.Getenv
    getExecutable = os.Executable
    osStat        = os.Stat
    osMkdirAll    = os.MkdirAll
)

func init() {
    initialize(runtime.GOOS)
}

func initialize(goos string) {
    if goos == "windows" {
        AppName += ".exe"
        CLIName += ".exe"
        // Logs, configs, downloads go to LOCALAPPDATA
        localAppData := getEnv("LOCALAPPDATA")
        AppDataDir = filepath.Join(localAppData, "Ollama")
        UpdateStageDir = filepath.Join(AppDataDir, "updates")
        AppLogFile = filepath.Join(AppDataDir, "app.log")
        ServerLogFile = filepath.Join(AppDataDir, "server.log")
        UpgradeLogFile = filepath.Join(AppDataDir, "upgrade.log")

        exe, err := getExecutable()
        if err != nil {
            slog.Warn("error discovering executable directory", "error", err)
            AppDir = filepath.Join(localAppData, "Programs", "Ollama")
        } else {
            AppDir = filepath.Dir(exe)
        }

        // Make sure we have PATH set correctly for any spawned children
        paths := strings.Split(getEnv("PATH"), ";")
        // Start with whatever we find in the PATH/LD_LIBRARY_PATH
        found := false
        for _, path := range paths {
            d, err := filepath.Abs(path)
            if err != nil {
                continue
            }
            if strings.EqualFold(AppDir, d) {
                found = true
            }
        }
        if !found {
            paths = append(paths, AppDir)

            pathVal := strings.Join(paths, ";")
            slog.Debug("setting PATH=" + pathVal)
            err := os.Setenv("PATH", pathVal)
            if err != nil {
                slog.Error(fmt.Sprintf("failed to update PATH: %s", err))
            }
        }

        // Make sure our logging dir exists
        _, err = osStat(AppDataDir)
        if errors.Is(err, os.ErrNotExist) {
            if err := osMkdirAll(AppDataDir, 0o755); err != nil {
                slog.Error(fmt.Sprintf("create ollama dir %s: %v", AppDataDir, err))
            }
        }
    } else if goos == "darwin" {
        // TODO
        AppName += ".app"
        // } else if runtime.GOOS == "linux" {
        // TODO
    }
}
