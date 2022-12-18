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
                res[i] = alpha *  losses[i] * activationDerivative(x[i]);
            }
            return res;
        }
    }
    internal class MatrixSigmoid : MatrixLayer
    {
        Matrix lastInput;
        
        double alpha = 2.0;
        public MatrixSigmoid()
        {
        }

        private double sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-alpha * x));
        }
        public Matrix forward(Matrix x)
        {
            lastInput = x;
            double[,] result = new double[x.n, x.m];
            for (int i = 0; i < x.n; i++)
            {
                for (int j = 0; j < x.m; j++)
                {
                    result[i, j] = sigmoid(x.data[i, j]);
                }
            }
            return new Matrix(result);
        }

        public Matrix backward(Matrix dout, double lr)
        {
            Matrix x = lastInput;
            double[,] result = new double[dout.n, dout.m];
            for (int i = 0; i < dout.n; i++)
            {
                for (int j = 0; j < dout.m; j++)
                {
                    result[i, j] = alpha * dout.data[i, j] * sigmoid(x.data[i, j]) * (1 - sigmoid(x.data[i, j]));
                }
            }
            return new Matrix(result);
        }
    
    }

}
