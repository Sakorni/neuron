using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1.ImageProcessor
{
    public class Processor
    {
        private const int maxMorseSignParts = 5;
        public double[] processImage(Bitmap original)
        {
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
            blobber.MinHeight = 2;
            blobber.MinWidth = 2;
            blobber.ObjectsOrder = AForge.Imaging.ObjectsOrder.XY;

            // Инвертируем 
            AForge.Imaging.Filters.Invert InvertFilter = new AForge.Imaging.Filters.Invert();
            InvertFilter.ApplyInPlace(uProcessed);

            blobber.ProcessImage(uProcessed);
            var rects = blobber.GetObjectsRectangles();
            var res = rects.Take(maxMorseSignParts).Select(x => x.Width * 1.0).ToList();
            while (res.Count < maxMorseSignParts)
            {
                res.Add(0);
            }

            return res.ToArray();
        }
    }
}
