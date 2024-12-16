#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QProcess>
#include <QFileDialog>
#include <QMessageBox>
#include <QDir>
#include <QTextStream>
#include <QInputDialog>
#include <QDateTime>
#include <Python.h>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , process(nullptr)
    , isCollecting(false)
{
    ui->setupUi(this);

    // 连接信号和槽
    connect(ui->newPatientButton, &QPushButton::clicked, this, &MainWindow::onNewPatientClicked);

    // 连接信号与槽
    connect(ui->startAllDataCollection, &QPushButton::clicked, this, &MainWindow::onStartAllDataCollection);
    connect(ui->stopAllDataCollection, &QPushButton::clicked, this, &MainWindow::onStopAllDataCollection);


}

MainWindow::~MainWindow()
{
    delete ui;
    if (process && process->state() == QProcess::Running) {
        process->terminate();  // 确保进程被终止
    }
    delete process;
}

void MainWindow::onNewPatientClicked()
{
    // 新建患者，生成唯一编号
    bool ok;
    QString patientName = QInputDialog::getText(this, tr("输入患者信息"), tr("请输入患者姓名："), QLineEdit::Normal, "", &ok);
    if (!ok || patientName.isEmpty()) {
        return;
    }

    // 自动生成患者编号（P0001、P0002...）
    QString patientId = generatePatientId();

    // 创建患者文件夹
    QString patientDir = QDir::currentPath() + "/patients/" + patientId;
    QDir().mkdir(patientDir);

    // 保存患者信息到文本文件
    QFile file(patientDir + "/patient_info.txt");
    if (file.open(QIODevice::WriteOnly)) {
        QTextStream out(&file);
        out << "姓名：" << patientName << "\n";
        out << "编号：" << patientId << "\n";
        // 保存其他信息
    }

    // 更新患者列表
    updatePatientList();
}

QString MainWindow::generatePatientId()
{
    // 自动生成患者编号，例如 P0001、P0002
    QDir dir(QDir::currentPath() + "/patients");
    int patientCount = dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot).size();
    return QString("P%1").arg(patientCount + 1, 4, 10, QChar('0'));
}

void MainWindow::updatePatientList()
{
    // 更新患者列表
    QDir dir(QDir::currentPath() + "/patients");
    QStringList patientNames = dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);

    ui->patientListWidget->clear();
    foreach (const QString &patientName, patientNames) {
        ui->patientListWidget->addItem(patientName);
    }
}


void MainWindow::onStartAllDataCollection()
{
    QString patientId = ui->patientListWidget->currentItem()->text();  // 获取选中的患者编号
    if (patientId.isEmpty()) {
        QMessageBox::warning(this, "警告", "请先选择患者！");
        return;
    }

    // 创建保存文件的路径
    QDir dir("D:/data/patients_data");  // 保存数据路径
    if (!dir.exists()) {
        dir.mkpath(".");  // 如果文件夹不存在，则创建它
    }

    QString patientDir = dir.absoluteFilePath(patientId);
    if (!QDir(patientDir).exists()) {
        QDir().mkdir(patientDir);  // 如果患者文件夹不存在，创建它
    }

    QString csvFilePath = patientDir + "/pulse_data.csv";  // 定义 CSV 文件路径

//    QString pythonScriptPath = "python";  // Python解释器路径
    QString pythonInterpreter = "D:/miniconda3/envs/falldetection/python.exe";

    // 启动 Python 进程
    if (process) {
        delete process;  // 确保之前的过程被删除
    }
    process = new QProcess(this);  // 创建 QProcess 对象
    QString scriptPath = "scripts/datacollector1.py";  // Python 脚本路径
    QStringList arguments;
    arguments << scriptPath << csvFilePath;  // 传递文件路径作为参数

//    // 连接信号和槽
//    connect(process, &QProcess::finished, this, &MainWindow::onProcessFinished);
    // 连接信号和槽
    connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &MainWindow::onProcessFinished);
    // 启动进程
    process->start(pythonInterpreter, arguments);
}

void MainWindow::onStopAllDataCollection()
{
    if (process && process->state() == QProcess::Running) {
        process->terminate();  // 终止正在运行的 Python 进程
        process->waitForFinished(1000);  // 等待进程结束，最多等待1秒
        if (process->state() == QProcess::Running) {
            process->kill();  // 如果进程仍未结束，强制杀死进程
        }
        QMessageBox::information(this, "信息", "数据采集已停止！");
    } else {
        QMessageBox::warning(this, "警告", "没有正在运行的数据采集进程！");
    }
}

void MainWindow::onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    if (exitStatus == QProcess::NormalExit) {
        QMessageBox::information(this, "信息", "数据采集完成！");
    } else {
        QMessageBox::critical(this, "错误", "数据采集失败！");
    }
    process->deleteLater();  // 删除进程对象
}
