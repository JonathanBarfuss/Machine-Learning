#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>

using namespace std;


//Struct to return the weight and bias for a linear equation
struct linearresults {
	double w;
	double b;
};

//Prototypes
linearresults train(vector<double> x, vector<double> y, int iterations, double lr);
double loss(vector<double> x, vector<double> y, double weight, double bias);
double predict(double x, double weight, double bias);
linearresults gradient(vector<double> x, vector<double> y, double weight, double bias);
linearresults g_train(vector<double> x, vector<double> y, int iterations, double lr);


//Main
int main() {

	//Gathering data--------------------------------------------------------------------------------------------
	vector<double> x;
	vector<double> y;
	char sx[5];
	char sy[5];

	ifstream myfile("data.txt");	//Open file
	myfile.seekg(3);	//Skip to 3rd character (because the first 2 are headings)

	//The following while loop extracts data from the txt file and gets stored into the vectors x and y
	while (myfile.get(sx, 5, '\t')) {	//Store characters into sx until 5 characters or a tab is reached
		myfile.get(sy, 5, '\n');		//Store characters into sy until 5 characters or a newline is reached
		cout << sx << " " << sy << endl;
		x.push_back(atof(sx));
		y.push_back(atof(sy));
	}
	myfile.close();		//Close file

	//Training model-----------------------------------------------------------------------------------------------
	linearresults model = g_train(x, y, 1000, .001);

	//Printing results---------------------------------------------------------------------------------------------
	cout << "The resulting model is " << model.w << "x + " << model.b << endl;


	return 0;
}

//Functions
linearresults train(vector<double> x, vector<double> y, int iterations, double lr) {
	linearresults answer;
	answer.w = answer.b = 0;

	for (int i = 0; i < iterations; i++) {
		double current_loss = loss(x, y, answer.w, answer.b);

		if (loss(x, y, answer.w - lr, answer.b) < current_loss) {
			answer.w -= lr;
		}
		else if (loss(x, y, answer.w + lr, answer.b) < current_loss) {
			answer.w += lr;
		}
		else if (loss(x, y, answer.w, answer.b - lr) < current_loss) {
			answer.b -= lr;
		}
		else if (loss(x, y, answer.w, answer.b + lr) < current_loss) {
			answer.b += lr;
		}
		else {
			return answer;
		}
	}

	throw std::runtime_error("Solution not found, try more iterations");
}

double loss(vector<double> x, vector<double> y, double weight, double bias) {
	//Creating and filling the predictions array
	vector<double> predictions;
	for (unsigned int i = 0; i < x.size(); i++) {
		predictions.push_back(predict(x[i], weight, bias));
	}
	//Calculating the average of the predictions - the answer
	double sum = 0;
	for(unsigned int i = 0; i < y.size(); i++) {
		sum += pow(predictions[i] - y[i], 2);
	}
	return sum / x.size();
}

double predict(double x, double weight, double bias) {
	return x * weight + bias;
}



//Gradient descent functions---------------------------------------------------------------------

linearresults gradient(vector<double> x, vector<double> y, double weight, double bias) {
	linearresults grad;
	vector<double> predictions;
	for (unsigned int i = 0; i < x.size(); i++) {
		predictions.push_back(predict(x[i], weight, bias));
	}

	//Calculating w gradient
	grad.w = 0;
	for (unsigned int i = 0; i < x.size(); i++) {
		grad.w += 2 * x[i] * (predictions[i] - y[i]);
	}

	//Calculating b gradient
	grad.b = 0;
	for (unsigned int i = 0; i < x.size(); i++) {
		grad.b += 2 * (predictions[i] - y[i]);
	}

	return grad;
}

linearresults g_train(vector<double> x, vector<double> y, int iterations, double lr) {
	linearresults answer;
	answer.w = answer.b = 0;

	for (int i = 0; i < iterations; i++) {
		//cout << "Loss is currently " << loss(x, y, answer.w, answer.b) << endl;
		linearresults grad = gradient(x, y, answer.w, answer.b);
		answer.w -= grad.w * lr;
		answer.b -= grad.b * lr;
	}

	return answer;
}
