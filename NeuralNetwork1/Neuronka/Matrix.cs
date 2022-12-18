﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1.Neuronka
{
    public class Matrix
    {
        public double[,] data;
        public int n, m; // n - rows, m - columns
        public Matrix(double[,] data) {
            this.data = data;
            n = data.GetLength(0);
            m = data.GetLength(1);
        }
        public Matrix(double[] data)
        {
            this.data = new double[1, data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                this.data[0,i] = data[i];
            }
            n = 1;
            m = data.Length;
        }
      
        public Matrix Transpose()
        {
            var res = new double[m,n];
            for (var i = 0; i < n; i++)
            {
                for(var j = 0; j < m; j++)
                {
                    res[j,i] = data[i,j];
                }
            }
            return new Matrix(res);
        }

        public double[] Sum(int axis)
        {
            if (axis == 0)
            {
                double[] result = new double[m];
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        result[i] += data[j, i];
                    }
                }
                return result;
            }
            else if (axis == 1)
            {
                double[] result = new double[n];
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < m; j++)
                    {
                        result[i] += data[i, j];
                    }
                }
                return result;
            }
            else
            {
                throw new Exception("Axis must be 0 or 1");
            }
        }


        public static Matrix operator /(Matrix m, double d)
        {
            var res = new double[m.n, m.m];
            Parallel.For(0, m.n, i =>
            {
                for (var j = 0; j < m.m; j++)
                {
                    res[i, j] = m.data[i, j] / d;
                }
            });
            return new Matrix(res);
        }
        public static Matrix operator +(Matrix a, double[] b)
        {
            if (a.n == b.Length)
            {

                double[,] result = new double[a.n, a.m];
                Parallel.For(0, a.n, i =>
                {



                    for (int j = 0; j < a.m; j++)
                    {
                        result[i, j] = a.data[i, j] + b[i];
                    }

                });

                return new Matrix(result);
            }else if( a.m == b.Length)
            {
                double[,] result = new double[a.n, a.m];
                Parallel.For(0, a.n, i =>
                {
                    for (int j = 0; j < a.m; j++)
                    {
                        result[i, j] = a.data[i, j] + b[j];
                    }
                });
                return new Matrix(result);
            }
            throw new Exception();
        }

        public static Matrix operator +(Matrix a, Matrix b)
        {
            if (a.n != b.n || a.m != b.m)
            {
                throw new Exception("Matrix dimensions must agree");
            }
            double[,] result = new double[a.n, a.m];
            Parallel.For(0, a.n, i =>
            {
                for (int j = 0; j < a.m; j++)
                {
                    result[i, j] = a.data[i, j] + b.data[i, j];
                }
            });
            return new Matrix(result);
        }

        public static Matrix operator -(Matrix a, Matrix b)
        {
            if (a.n != b.n || a.m != b.m)
            {
                throw new Exception("Matrix dimensions must agree");
            }
            double[,] result = new double[a.n, a.m];
            Parallel.For(0, a.n, i =>
            {
                for (int j = 0; j < a.m; j++)
                {
                    result[i, j] = a.data[i, j] - b.data[i, j];
                }
            });
            return new Matrix(result);
        }

        public static Matrix operator *(Matrix a, double b)
        {
            double[,] result = new double[a.n, a.m];
            Parallel.For(0, a.n, i =>
            { 
                for (int j = 0; j < a.m; j++)
                {
                    result[i, j] = a.data[i, j] * b;
                }
            });
            return new Matrix(result);
        }

        public static Matrix operator *(Matrix a, Matrix b)
        {
            var res = new double[a.n, b.m];
            Parallel.For(0, a.n, i =>
            {
                Parallel.For(0, b.m, j =>
                {
                    for (var k = 0; k < b.n; k++)
                    {
                        res[i, j] += a.data[i, k] * b.data[k, j];
                    }
                });
            });
            return new Matrix(res);
        }


        public static double[] operator *(double[] a, Matrix b)
        {
            if(a.Length != b.n)
            {
                throw new Exception("Invalid dimensions");
            }
            double[] result = new double[b.m];
            Parallel.For(0, b.m, i =>

            {
                result[i] = 0;
                for (int j = 0; j < b.n; j++)
                {
                    result[i] += b.data[j, i] * a[j];
                }
            });
            return result;
        }
        public static double[] operator *(Matrix b, double[] a)
        {
            if (a.Length != b.m)
            {
                throw new Exception("Invalid dimensions");
            }
            double[] result = new double[b.n];
            Parallel.For(0, b.n, i =>
            {
                result[i] = 0;
                for (int j = 0; j < b.m; j++)
                {
                    result[i] += b.data[i, j] * a[j];
                }
            });
            return result;
        }

    }

}
