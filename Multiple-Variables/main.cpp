#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
#include "matrix.h"

using namespace std;

//Prototypes
double loss(QSMatrix<double> x, QSMatrix<double> y, QSMatrix<double> weight);
QSMatrix<double> predict(QSMatrix<double> x, QSMatrix<double> weight);
QSMatrix<double> gradient(QSMatrix<double> x, QSMatrix<double> y, QSMatrix<double> weight);
QSMatrix<double> train(QSMatrix<double> x, QSMatrix<double> y, int iterations, double lr);


//Main
int main() {

	//Gathering data--------------------------------------------------------------------------------------------
	QSMatrix<double> x(18, 5, 1);
	QSMatrix<double> y(18, 1, 0);
	char sx[5];
	char sa[5];
	char sb[5];
	char sc[5];
	char sy[5];

	ifstream myfile("data3.txt");	//Open file
	myfile.seekg(9);	//Skip to 9th character (because the first 8 are headings)

	int i = 0;
	//The following while loop extracts data from the txt file and gets stored into the vectors x and y
	while (myfile.getline(sx, 5, '\t')) {	//Store characters into sx until 5 characters or a tab is reached
		myfile.getline(sa, 5, '\t');
		myfile.getline(sb, 5, '\t');
		myfile.getline(sc, 5, '\t');
		myfile.getline(sy, 5, '\n');		//Store characters into sy until 5 characters or a newline is reached
		
		x(i, 0) = (atof(sx));
		x(i, 1) = (atof(sa));
		x(i, 2) = (atof(sb));
		x(i, 3) = (atof(sc));
		y(i, 0) = (atof(sy));
		i++;
	}
	myfile.close();		//Close file

	//Training model-----------------------------------------------------------------------------------------------
	QSMatrix<double> model(x.get_cols(), 1, 0);
	model = train(x, y, 15000, .00005);

	//Printing results---------------------------------------------------------------------------------------------
	cout << "The resulting model is " << model(0, 0) << "x + " << model(1, 0) << "a + " << model(2, 0) << "b + " << model(3, 0) << "c + " << model(4, 0) << endl;


	return 0;
}

//Functions-------------------------------------------------------------------------------------------------------------------

double loss(QSMatrix<double> x, QSMatrix<double> y, QSMatrix<double> weight) {
	//Creating and filling the predictions matrix
	QSMatrix<double> predictions(y.get_rows(), y.get_cols(), 0);
	predictions = predict(x, weight);

	//Calculating the average of the predictions minus the answer
	double sum = 0;
	for (unsigned int i = 0; i < y.get_rows(); i++) {
		sum += pow(predictions(i,0) - y(i,0), 2);
	}
	return sum / x.get_rows();
}

QSMatrix<double> predict(QSMatrix<double> x, QSMatrix<double> weight) {
	return x * weight;
}

QSMatrix<double> gradient(QSMatrix<double> x, QSMatrix<double> y, QSMatrix<double> weight) {
	QSMatrix<double> grad(weight.get_rows(), weight.get_cols(), 0);
	QSMatrix<double> predictions(y.get_rows(), 1, 0);
	
	predictions = predict(x, weight);

	//Formatting the x matrix
	QSMatrix<double> xt(x.transpose());

	//Calculating w gradient
	for (unsigned int i = 0; i < x.get_rows(); i++) {
		grad += xt * (predict(x, weight) - y) * 2;
	}

	return grad / x.get_rows();
}

QSMatrix<double> train(QSMatrix<double> x, QSMatrix<double> y, int iterations, double lr) {
	//Creating a weight matrix
	QSMatrix<double> w(x.get_cols(), 1, 0);

	//Each iteration changes the value in w to make a more accurate model. Gradient descent decides how they are changed.
	for (int i = 0; i < iterations; i++) {
		QSMatrix<double> grad(x.get_cols(), 1, 0);
		grad = gradient(x, y, w);
		w -= grad * lr;
		cout << i << ": " << w(0, 0) << " " << w(1, 0) << " " << w(2, 0) << " " << w(3, 0) << " " << w(4, 0) << endl;	//Printing each iteration
	}

	return w;
}