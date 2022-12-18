using System;

namespace NeuralNetwork1.Neuronka
{

    public class Simple : Layer
    {
        public double[] backward(double[] losses, double learningRate)
        {
            return losses;
        }

        public double[] forward(double[] input)
        {
            return input;
        }
    }
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
            var dx = weights * losses;
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
    }
    public class LinearMatrix : MatrixLayer
    {
        int input_size;
        int output_size;

        Matrix weights;
        double[] b;
        Matrix lastInput;
        public LinearMatrix(int input_size, int output_size)
        {
            this.input_size = input_size;
            this.output_size = output_size;

            this.b = new double[this.output_size];
            init_weights();
        }

        private void init_weights(double lower_bound = -0.005, double upper_bound = 0.005)
        {
            Random random = new Random();
            double[,] w = new double[input_size, output_size];
            for (int i = 0; i < input_size; i++)
            {
                for (int j = 0; j < output_size; j++)
                {
                    w[i, j] = random.NextDouble() * (upper_bound - lower_bound) + lower_bound;
                }
            }

            for (int i = 0; i < output_size; i++)
            {
                b[i] = 0;
            }
            weights = new Matrix(w);
        }

        Matrix MatrixLayer.forward(Matrix x)
        //returns Wx + b
        {
            lastInput = x;
            return (x * weights) + b;
        }

        Matrix MatrixLayer.backward(Matrix dout, double lr)
        {
            Matrix x = lastInput;
            double[] db = dout.Sum(0);
            Matrix dx = dout * weights.Transpose();
            Matrix dW = x.Transpose() * dout;

            weights = weights - dW * lr;

            for (int i = 0; i < output_size; i++)
            {
                b[i] -= db[i] * lr;
            }

            return dx;
        }
    }
}


