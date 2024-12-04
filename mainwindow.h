#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QListWidget>
#include <QProcess>
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QInputDialog>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    // 构造函数和析构函数
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    // 槽函数，用于响应按钮点击事件
    void onNewPatientClicked(); // 新建患者
    void onStartFingerCollection(); // 启动指夹仪数据采集
    void onStartWristCollection(); // 启动手腕脉搏仪数据采集
    void onStartCameraCollection(); // 启动摄像头数据采集
    // 开始数据采集
    void onStartAllDataCollection();
    // 停止数据采集
    void onStopAllDataCollection();
    // 处理 Python 进程结束的信号
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);


private:
    // UI 指针
    Ui::MainWindow *ui;
    QProcess *process;  // 用于启动和控制 Python 脚本的进程
    bool isCollecting;  // 标志是否正在采集数据
    // 私有函数，生成患者编号
    QString generatePatientId();
    // 私有函数，更新患者列表
    void updatePatientList();
    // 私有函数，启动数据采集（指夹仪、手腕脉搏仪、摄像头）
    void startDataCollection(const QString &scriptPath, const QString &patientId, QLabel *statusLabel);
};

#endif // MAINWINDOW_H
