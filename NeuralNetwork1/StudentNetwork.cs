using NeuralNetwork1.Neuronka;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        private double learningRate = 0.4;
        private int trainLimit = 200;
        List<Layer> layers;
        int[] structure;
        string backupFile = "..\\..\\backup.txt";


        private double[] LossFunction(double[] result, double[] expected, out double totalLoss)
        {
            totalLoss = 0.0;
            var deriv = new double[result.Length];
            for (var i = 0; i < result.Length; i++)
            {
                totalLoss += Math.Pow((expected[i] - result[i]), 2) / expected.Length;
                deriv[i] = ((2 * (result[i] - expected[i])) / result.Length * 0.01);  //  dC / dOut

            }
            // totalLoss /= expected.Length;
            return deriv;
        }

        public StudentNetwork(int[] structure)
        {
            this.structure = structure;
            layers = new List<Layer>();
            for (int i = 0; i < structure.Length - 1; i++)
            {
                layers.Add(new Linear(structure[i], structure[i + 1]));
                layers.Add(new Sigmoid());
            }

        }

        private double[] forward(double[] input)
        {
            double[] result = input;
            foreach (var layer in layers)
            {
                result = layer.forward(result);
            }
            return result;
        }

        private void backward(double[] data)
        {
            Object tData = data;
            for (var i = layers.Count - 1; i >= 0; i--)
            {
                tData = layers[i].backward((double[])tData, learningRate);
            }
        }
        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int iteration = 0;
            var error = 1e6;
            Object res = null;
            while (error > acceptableError && iteration < trainLimit)
            {
                res = trainSample(sample, out error);
                iteration++;
            }
            var max = -1e6;
            int answer = -1;
            for (int i = 0; i < ((double[])res).Length; i++)
            {
                if (((double[])res)[i] > max)
                {
                    answer = i;
                    max = ((double[])res)[i];
                }
            }
            return answer;
        }

        private double[] trainSample(Sample sample, out double error)
        {
            error = 0.0;
            var res = forward(sample.input);
            var dx = LossFunction(res, sample.Output, out error);
            backward(dx);
            return res;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            var start = DateTime.Now;
            var totalSamples = epochsCount * samplesSet.Count;
            var processed = 0;
            var error = 0.0;
            var avgError = 0.0;
            for (int e = 0; e < epochsCount; e++)
            {
                var sumErr = 0.0;
                for (int i = 0; i < samplesSet.Count; i++)
                {
                    var err = 0.0;
                    var sample = samplesSet[i];
                    trainSample(sample, out err);
                    sumErr += err;
                    processed++;
                }
                OnTrainProgress(1.0 * processed / totalSamples,
                            sumErr / samplesSet.Count, DateTime.Now - start);
                if (sumErr <= acceptableError)
                {
                    OnTrainProgress(1.0, sumErr, DateTime.Now - start);
                    return sumErr;
                }
                error += sumErr;
            }
            avgError = error / totalSamples;
            OnTrainProgress(1.0, avgError, DateTime.Now - start);
            return avgError;
        }

        protected override double[] Compute(double[] input)
        {
            return forward(input);
        }
        override public void Backup() 
        {
            using (StreamWriter writer = new StreamWriter(backupFile))
            {
                writer.WriteLine(string.Join(";",structure.Select(f => f.ToString())));
                foreach(var layer in layers)
                {
                    var data = layer.inners();
                    if(data.Length == 0)
                    {
                        continue;
                    }
                    for(int i = 0; i < data.GetLength(0); i++)
                    {
                        for (int j = 0; j < data.GetLength(1); j++)
                        {
                            writer.Write(data[i, j]);
                            if( j+1 < data.GetLength(1))
                                writer.Write(" ");
                        }
                        writer.WriteLine();
                    }
                }
            }
        }
        override public void LoadBackup()
        {
            using (StreamReader reader = new StreamReader(backupFile))
            {
                structure = reader.ReadLine().Split(';').Select(f => int.Parse(f)).ToArray();
                var layersCount = structure.Length;
                var layers = new List<Layer>();
                for(int i = 0; i < layersCount-2; i++)
                {
                    var n = structure[i];
                    var m = structure[i + 1];
                    var data = new double[n, m];
                    for (int x = 0; x < n; x++)
                    {
                        var line = reader.ReadLine().Split(' ');
                        for (int y = 0; y < m; y++)
                        {
                            data[x, y] = double.Parse(line[y]);
                        }
                        layers.Add(new Linear(data));
                        layers.Add(new Sigmoid());
                    }
                }
            }
        }

    }
}