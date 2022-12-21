using Accord.Math;
using Accord.Statistics.Kernels;
using System;
using System.Linq;

namespace NeuralNetwork1.Neuronka
{
    public class Linear : Layer
    {
        int inputSize;
        int outputSize;
        Matrix weights;
        double[] lastInput;
        double[] bias;

        public Linear(int inputSize, int outputSize)
        {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            bias = new double[outputSize];
            var wData = new double[inputSize, outputSize];
            var random = new Random();
            for(var i = 0; i < outputSize; i++)
            {
                bias[i] = 1;
                for (var j = 0; j < inputSize; j++)
                {
                    wData[j,i] = random.NextDouble() *2.0 - 1.0;
                }
            }
            weights = new Matrix(wData);
        }

        public double[] backward(double[] losses, double learningRate)
        {
            var l = new Matrix(losses);
            var x = new Matrix(lastInput);
            var dx = losses * weights.Transpose();
            var dw = x.Transpose() * l;
            this.weights -= (dw * learningRate);
            
            for(var i = 0; i < bias.Length; i++)
            {
                  bias[i] -= (losses[i] * learningRate);
            }

            return dx;
        }

        // y = (x * w) + b; Neuron activation is "тождественная" function
        public double[] forward(double[] input)
        {
            var t = input * weights;
            for(var i = 0; i < outputSize; i++)
            {
                t[i] += bias[i];
            }
            lastInput= input;
            return t;
        }

        public double[,] inners()
        {
            var extended = this.weights.data.ToJagged().ToList();
            extended.Add(bias);
            return extended.ToArray().ToMatrix();
        }
        Layer Layer.FromSave(double[,] data)
        {
            return new Linear(data);
        }

        public Linear(double[,] data)
        {
            var t = data.ToJagged().ToList();
            bias = t.Last();
            t.RemoveAt(t.Count-1);
            weights = new Matrix(t.ToArray().ToMatrix());
            inputSize = weights.n;
            outputSize = weights.m;
        }
    }
}


