using NeuralNetwork1.Neuronka;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        private double learningRate = 1;
        private int trainLimit = 100;
        List<Layer> layers;

        private double[] LossFunction(double[] result, double[] expected, out double totalLoss, double samples = 100) 
        {
            totalLoss= 0.0;
            var deriv = new double[result.Length];
            for (var i = 0; i < result.Length; i++)
            {
                totalLoss += Math.Pow((expected[i] - result[i]), 2) / expected.Length;
                deriv[i] =( (2 * (result[i] - expected[i])) / result.Length) / samples;  //  dC / dOut

            }
            // totalLoss /= expected.Length;
            return deriv;
        }

        public StudentNetwork(int[] structure)
        {
            layers = new List<Layer>();
            layers.Add(new Sigmoid());

            for (int i = 0; i < structure.Length -1; i++)
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
            for (var i = layers.Count-1; i >= 0; i--)
            {
                tData = layers[i].backward((double[])tData, learningRate);
            }
        }
        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int iteration = 0;
            var error = 1e6;
            Object res = null;
            while(error > acceptableError && iteration < trainLimit )
            {
                res = trainSample(sample, out error);
                iteration++;
            }
            var max = -1e6;
            int answer = -1;
            for(int i = 0; i < ((double[])res).Length; i++)
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
            for(int e = 0; e < epochsCount; e++)
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

    }

    public class MatrixStudentNetwork : BaseNetwork
    {
        int[] structure;
        double lr = 1.0;
        List<MatrixLayer> layers;
        public Stopwatch stopWatch = new Stopwatch();
        public MatrixStudentNetwork(int[] structure)
        {
            this.structure = structure;
            this.layers = new List<MatrixLayer>();
            for (int i = 0; i < structure.Length - 1; i++)
            {
                this.layers.Add(new LinearMatrix(structure[i], structure[i + 1]));
                this.layers.Add(new MatrixSigmoid());
            }

        }
        public List<object> MSE_forward(Matrix x, Matrix y)
        //y  - метки классов в батче
        {
            double loss = 0;
            double[,] dx = new double[x.n, x.m];

            for (int i = 0; i < x.n; i++)
            {
                for (int j = 0; j < x.m; j++)
                {
                    loss += Math.Pow(x.data[i, j] - y.data[i, j], 2) / x.m;
                    dx[i, j] = 2 * (x.data[i, j] - y.data[i, j]) / x.m;
                }
            }
            loss /= x.n;
            Matrix dX = new Matrix(dx);
            dX = dX / x.n;
            return new List<object> { loss, dX };
        }

        public Matrix Forward(Matrix x)
        {
            Matrix result = x;
            for (int i = 0; i < this.layers.Count; i++)
            {
                result = this.layers[i].forward(result);
            }
            return result;
        }

        public void Backward(Matrix dout)
        {
            for (int i = this.layers.Count - 1; i >= 0; i--)
            {
                dout = this.layers[i].backward(dout, this.lr);
            }
        }

        public double Run(Matrix x, Matrix y)
        {
            //x = x.normilize();
            Matrix pred = Forward(x);
            List<object> res = MSE_forward(pred, y);
            double loss = (double)res[0];
            Matrix dX = (Matrix)res[1];

            this.Backward(dX);

            return loss;
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int iters = 1;
            double[,] ax = new double[1, sample.input.Length];
            for (int i = 0; i < sample.input.Length; i++)
            {
                ax[0, i] = sample.input[i];
            }
            double[,] ay = new double[1, sample.Output.Length];
            for (int i = 0; i < sample.Output.Length; i++)
            {
                ay[0, i] = sample.Output[i];
            }
            Matrix x = new Matrix(ax);
            Matrix y = new Matrix(ay);
            double loss = this.Run(x, y);
            while (loss > acceptableError)
            {
                loss = this.Run(x, y);
                iters++;
            }

            return iters;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            double[,] inputs = new double[samplesSet.Count, samplesSet[0].input.Length];
            double[,] outputs = new double[samplesSet.Count, samplesSet[0].Output.Length];

            for (int i = 0; i < samplesSet.Count; ++i)
            {
                for (int j = 0; j < samplesSet[i].input.Length; ++j)
                {
                    inputs[i, j] = samplesSet[i].input[j];
                }

                for (int j = 0; j < samplesSet[i].Output.Length; ++j)
                {
                    outputs[i, j] = samplesSet[i].Output[j];
                }
            }
            Matrix x = new Matrix(inputs);
            Matrix y = new Matrix(outputs);

            int epoch_to_run = 0;
            double error = double.PositiveInfinity;

            StreamWriter errorsFile = File.CreateText("errors.csv");

            stopWatch.Restart();

            while (epoch_to_run < epochsCount && error > acceptableError)
            {
                epoch_to_run++;
                error = this.Run(x, y);
#if DEBUG
                errorsFile.WriteLine(error);
#endif
                OnTrainProgress((epoch_to_run * 1.0) / epochsCount, error, stopWatch.Elapsed);
            }

#if DEBUG
            errorsFile.Close();
#endif
            OnTrainProgress(1.0, error, stopWatch.Elapsed);

            stopWatch.Stop();

            return error;
        }

        protected override double[] Compute(double[] input)
        {
            double[,] ax = new double[1, input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                ax[0, i] = input[i];
            }

            Matrix x = new Matrix(ax);
            Matrix y = this.Forward(x);
            double[] result = new double[y.m];
            for (int i = 0; i < y.m; i++)
            {
                result[i] = y.data[0, i];
            }

            return result;
        }
    }

}
