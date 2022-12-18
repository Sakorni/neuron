using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1.Neuronka
{
    internal interface Layer
    {
        double[] forward(double[] input);
        double[] backward(double[] losses, double learningRate);
    }
    internal interface MatrixLayer
    {
        Matrix forward(Matrix input);
        Matrix backward(Matrix dout, double learningRate);
    }
}
