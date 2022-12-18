using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace NeuralNetwork1
{
    static class Program
    {
        /// <summary>
        /// Главная точка входа для приложения.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new NeuralNetworksStand(new Dictionary<string, Func<int[], BaseNetwork>>
            {
                // Тут можно добавить свои нейросети
                {"Студентческий персептрон", structure => new StudentNetwork(structure)},
                {"Matrix", structure => new MatrixStudentNetwork(structure)},
                {"Accord.Net Perseptron", structure => new AccordNet(structure)},

            }));
        }
    }
}