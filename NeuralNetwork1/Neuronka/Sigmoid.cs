using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1.Neuronka
{
    internal class Sigmoid : Layer
    {
        double alpha = 0.7;
        double[] lastInput;
        private double activation(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-alpha * x));
        }
        private double activationDerivative(double x)
        {
            var a = activation(x);
            return a*(1 - a);
        }
        public double[] forward(double[] input)
        {
            var result = new double[input.Length];
            for(int i = 0; i < result.Length; i++)
            {
                result[i] = activation(input[i]);
            }
            lastInput = input;
            return result;
        }

        public double[] backward(double[] losses, double learningRate)
        {
            var x = lastInput;
            var res = new double[losses.Length];
            for (int i = 0; i < losses.Length; i++)
            {
                res[i] = learningRate *  losses[i] * activationDerivative(x[i]);
            }
            return res;
        }

        public double[,] inners()
        {
            return new double[0,0];
        }

        public Layer FromSave(double[,] data)
        {
            return new Sigmoid();
        }
    }
}
