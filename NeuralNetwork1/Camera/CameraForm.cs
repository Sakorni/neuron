using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using AForge.WindowsForms;
using AForge.Video;
using AForge.Video.DirectShow;
using NeuralNetwork1.Dataset;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.Button;
using NeuralNetwork1.Properties;

namespace NeuralNetwork1.Camera
{
    delegate void FormUpdateDelegate();

    public partial class MainForm : Form
    {
        /// <summary>
        /// Класс, реализующий всю логику работы
        /// </summary>
        private Controller controller = null;

        /// <summary>
        /// Событие для синхронизации таймера
        /// </summary>
        private AutoResetEvent evnt = new AutoResetEvent(false);

        /// <summary>
        /// Список устройств для снятия видео (веб-камер)
        /// </summary>
        private FilterInfoCollection videoDevicesList;

        /// <summary>
        /// Выбранное устройство для видео
        /// </summary>
        private IVideoSource videoSource;

        /// <summary>
        /// Таймер для измерения производительности (времени на обработку кадра)
        /// </summary>
        private Stopwatch sw = new Stopwatch();

        /// <summary>
        /// Таймер для обновления объектов интерфейса
        /// </summary>
        System.Threading.Timer updateTmr;

        /// <summary>
        /// Функция обновления формы, тут же происходит анализ текущего этапа, и при необходимости переключение на следующий
        /// Вызывается автоматически - это плохо, надо по делегатам вообще-то
        /// </summary>
        private void UpdateFormFields()
        {
            //  Проверяем, вызвана ли функция из потока главной формы. Если нет - вызов через Invoke
            //  для синхронизации, и выход
            if (statusLabel.InvokeRequired)
            {
                this.Invoke(new FormUpdateDelegate(UpdateFormFields));
                return;
            }

            sw.Stop();
            ticksLabel.Text = "Тики : " + sw.Elapsed.ToString();
            originalImageBox.Image = controller.GetOriginalImage();
        }

        /// <summary>
        /// Обёртка для обновления формы - перерисовки картинок, изменения состояния и прочего
        /// </summary>
        /// <param name="StateInfo"></param>
        public void Tick(object StateInfo)
        {
            UpdateFormFields();
            return;
        }
        BaseNetwork net;
        public MainForm(BaseNetwork net)
        {
            this.net = net;
            InitializeComponent();
            // Список камер получаем
            videoDevicesList = new FilterInfoCollection(FilterCategory.VideoInputDevice);
            foreach (FilterInfo videoDevice in videoDevicesList)
            {
                cmbVideoSource.Items.Add(videoDevice.Name);
            }
            if (cmbVideoSource.Items.Count > 0)
            {
                cmbVideoSource.SelectedIndex = 0;
            }
            else
            {
                MessageBox.Show("А нет у вас камеры!", "Ошибочка", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            controller = new Controller(new FormUpdateDelegate(UpdateFormFields));
            //            updateTmr = new System.Threading.Timer(Tick, evnt, 500, 100);
        }

        private void video_NewFrame(object sender, NewFrameEventArgs eventArgs)
        {
            //  Время засекаем
            sw.Restart();

            //  Отправляем изображение на обработку, и выводим оригинал (с раскраской) и разрезанные изображения
            if (controller.Ready)

#pragma warning disable CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed
                controller.ProcessImage((Bitmap)eventArgs.Frame.Clone());
#pragma warning restore CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed

            //  Это выкинуть в отдельный поток!
            //  И отдать делегат? Или просто проверять значение переменной?
            //  Тут хрень какая-то

            //currentState = Stage.Thinking;
            //sage.solveState(processor.currentDeskState, 16, 7);
        }

        private void btnStart_Click(object sender, EventArgs e)
        {
            if (videoSource == null)
            {
                var vcd = new VideoCaptureDevice(videoDevicesList[cmbVideoSource.SelectedIndex].MonikerString);
                vcd.VideoResolution = vcd.VideoCapabilities[resolutionsBox.SelectedIndex];
                Debug.WriteLine(vcd.VideoCapabilities[1].FrameSize.ToString());
                Debug.WriteLine(resolutionsBox.SelectedIndex);
                videoSource = vcd;
                videoSource.NewFrame += new NewFrameEventHandler(video_NewFrame);
                videoSource.Start();
                StartButton.Text = "Стоп";
                controlPanel.Enabled = true;
                cmbVideoSource.Enabled = false;
            }
            else
            {
                videoSource.SignalToStop();
                if (videoSource != null && videoSource.IsRunning && originalImageBox.Image != null)
                {
                    originalImageBox.Image.Dispose();
                }
                videoSource = null;
                StartButton.Text = "Старт";
                controlPanel.Enabled = false;
                cmbVideoSource.Enabled = true;
            }
        }

        private void tresholdTrackBar_ValueChanged(object sender, EventArgs e)
        {
            controller.settings.threshold = (byte)tresholdTrackBar.Value;
            controller.settings.differenceLim = (float)tresholdTrackBar.Value / tresholdTrackBar.Maximum;
        }

        private void borderTrackBar_ValueChanged(object sender, EventArgs e)
        {
            controller.settings.border = borderTrackBar.Value;
        }

        private void marginTrackBar_ValueChanged(object sender, EventArgs e)
        {
            controller.settings.margin = marginTrackBar.Value;
        }

        private void MainForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (updateTmr != null)
                updateTmr.Dispose();

            //  Как-то надо ещё робота подождать, если он работает

            if (videoSource != null && videoSource.IsRunning)
            {
                videoSource.SignalToStop();
            }
        }

        private void MainForm_KeyDown(object sender, KeyEventArgs e)
        {
            switch (e.KeyCode)
            {
                case Keys.W: controller.settings.decTop(); Debug.WriteLine("Up!"); break;
                case Keys.S: controller.settings.incTop(); Debug.WriteLine("Down!"); break;
                case Keys.A: controller.settings.decLeft(); Debug.WriteLine("Left!"); break;
                case Keys.D: controller.settings.incLeft(); Debug.WriteLine("Right!"); break;
                case Keys.Q: controller.settings.border++; Debug.WriteLine("Plus!"); break;
                case Keys.E: controller.settings.border--; Debug.WriteLine("Minus!"); break;
            }
        }

        private void cmbVideoSource_SelectionChangeCommitted(object sender, EventArgs e)
        {
            var vcd = new VideoCaptureDevice(videoDevicesList[cmbVideoSource.SelectedIndex].MonikerString);
            resolutionsBox.Items.Clear();
            for (int i = 0; i < vcd.VideoCapabilities.Length; i++)
                resolutionsBox.Items.Add(vcd.VideoCapabilities[i].FrameSize.ToString());
            resolutionsBox.SelectedIndex = 0;
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            controller.settings.processImg = checkBox1.Checked;
        }

        private void originalImageBox_Click(object sender, EventArgs e)
        {

        }

        private void ProcessButton_Click(object sender, EventArgs e)
        {
            var original = (Bitmap)controller.GetOriginalImage().Clone();
            // А НЕЧЕГО В КРАЙ (краснодарский (это по словам Михаила, он еще посмеялся так неприятно))СУВАТЬ МОРЗЯНКУ
            int xborder = original.Width / 10;
            int yborder = original.Height / 10;
            //Rectangle cropRect = new Rectangle(xborder, yborder, original.Width - xborder, original.Height - yborder);
            Rectangle cropRect = new Rectangle(0, 0, original.Width, original.Height);

            //  Теперь всю эту муть пилим в обработанное изображение
            AForge.Imaging.Filters.Crop cropFilter = new AForge.Imaging.Filters.Crop(cropRect);
            var uProcessed = cropFilter.Apply(AForge.Imaging.UnmanagedImage.FromManagedImage(original));
            AForge.Imaging.Filters.Grayscale grayFilter = new AForge.Imaging.Filters.Grayscale(0.2125, 0.7154, 0.0721);
            uProcessed = grayFilter.Apply(uProcessed);

            //  Пороговый фильтр применяем. Величина порога берётся из настроек, и меняется на форме
            // Нет не меняется. Здесь мы "выдавливаем" изображение.
            AForge.Imaging.Filters.BradleyLocalThresholding threshldFilter = new AForge.Imaging.Filters.BradleyLocalThresholding();
            threshldFilter.PixelBrightnessDifferenceLimit = 0.13f;
            threshldFilter.ApplyInPlace(uProcessed);

            AForge.Imaging.BlobCounter blobber = new AForge.Imaging.BlobCounter();
            var minHeight = 2;
            var minWidth = 2;
            var maxHeight = original.Height - 30;
            var maxWidth = original.Width - 30;
            blobber.ObjectsOrder = AForge.Imaging.ObjectsOrder.XY;
            // Инвертируем 
            AForge.Imaging.Filters.Invert InvertFilter = new AForge.Imaging.Filters.Invert();
            InvertFilter.ApplyInPlace(uProcessed);
            blobber.ProcessImage(uProcessed);
            var rects = blobber.GetObjectsRectangles().Where(
                x => minHeight < x.Height && x.Height < maxHeight &&
                minWidth < x.Width && x.Width < maxWidth).ToArray();
            var maxRectWidth = rects.Max(x => x.Width);
            var maxRectWidthY = rects.Where(x => x.Width == maxRectWidth).First().Y;
            var res = rects.Where(x => Math.Abs(maxRectWidthY - x.Y) < 30).Take(5).Select(x => x.Width * 1.0).ToList();
            while (res.Count < 5)
            {
                res.Add(0);
            }

            var resImg = original;
            Graphics g = Graphics.FromImage(original);
            Pen p = new Pen(Color.Red);
            p.Width = 1;
            g.DrawRectangles(p, rects);

            processedImgBox.Image = resImg;
            
        }

        private void PlayButton_Click(object sender, EventArgs e)
        {
            var pr = new NeuralNetwork1.ImageProcessor.Processor();
            var prikol = pr.processImage((Bitmap)controller.GetOriginalImage().Clone());
            var punchline = net.Predict(new Sample(prikol, 10));
            MessageBox.Show(MorseWrapper.MorseToString(punchline), "HAHA", MessageBoxButtons.OK);
        }
    }
}
