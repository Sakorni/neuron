using Accord.Math;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using NeuralNetwork1.ImageProcessor;
using static NeuralNetwork1.Dataset.MorseWrapper;
namespace NeuralNetwork1.Dataset
{
    public class MorseWrapper {
        public enum Morse
        {
            a, b, v, g, d, e, k, o, p, c, UNDEF=-1
        };
        public static string MorseToString(Morse symbol)
        {
            if(symbol == Morse.UNDEF)
            {
                return "Кря-кря-кря";
            }
            switch (symbol)
            {
                case Morse.a:
                    return "а";
                case Morse.b:
                    return "б";
                case Morse.v:
                    return "в";
                case Morse.g:
                    return "г";
                case Morse.d:
                    return "д";
                case Morse.e:
                    return "е";
                case Morse.k:
                    return "к";
                case Morse.o:
                    return "о";
                case Morse.p:
                    return "п";
                case Morse.c:
                    return "с";

            }
            return ((int)symbol).ToString();
        }
        public static Morse FromInt(int i)
        {
            switch (i)
            {
                case 0: return Morse.a;
                case 1: return Morse.b;
                case 2: return Morse.v;
                case 3: return Morse.g;
                case 4: return Morse.d;
                case 5: return Morse.e;
                case 6: return Morse.k;
                case 7: return Morse.o;
                case 8: return Morse.p;
                case 9: return Morse.c;
                default: return Morse.UNDEF;
            }
        }
    }

    public class Processor
    {

        NeuralNetwork1.ImageProcessor.Processor imgProcessor;
        /// <summary>
        /// Рандом нужен нам для того, чтобы избежать ситуации, когда часть сэмплов мы игнорируем
        /// </summary>
        private Random rnd;
        private string dataSetDirPath = "..\\..\\Dataset\\dataset";
        /// <summary>
        /// Ключ - символ морзянки, значение - все файлы, которые содержат sample для этого символа
        /// </summary>
        private Dictionary<Morse, List<Bitmap>> samples;

        public int PictureSize = 200;
        public int SymbolCount = 2;

        /// <summary>
        /// Инициализируем и считываем все данные из директории датасета
        /// </summary>
        public Processor()
        {
            imgProcessor = new ImageProcessor.Processor();
            rnd = new Random();
            samples = new Dictionary<Morse, List<Bitmap>>();
            for (int i = 0; i < 10; i++)
            {
                var symbol = FromInt(i);
                DirectoryInfo dir = new DirectoryInfo(dataSetDirPath + $"\\{MorseToString(symbol)}");
                samples[symbol] = dir.GetFiles("*.jpg").Select(f => new Bitmap(f.FullName)).ToList();
                //System.Console.WriteLine(samples[FromInt(i)].ToString());
            }
        }
        public SamplesSet BuildSampleSet(int count)
        {
            var res = new SamplesSet();
            for (int i = 0; i < SymbolCount; i++)
            {
                Morse symbol = FromInt(i);
                var sset = samples[symbol];
                for (int j = 0; j < count; j++)
                {
                    var pic = sset[rnd.Next(sset.Count)];
                    var input = fillInput(pic);
                    res.AddSample(new Sample(input, SymbolCount, symbol));
                }
            }
            res.samples.Shuffle();
            return res;
        }
        private double[] fillInput(Bitmap pic)
        {
            return imgProcessor.processImage(pic);
        }


        public Tuple<Sample, Bitmap> SingleSample()
        {
            var symbol = FromInt(rnd.Next(SymbolCount));
            var sset = samples[symbol];
            var pic = sset[rnd.Next(sset.Count)];
            var input = fillInput(pic);
            return new Tuple<Sample, Bitmap>(new Sample(input, SymbolCount, symbol), pic);
        }
    }
}
