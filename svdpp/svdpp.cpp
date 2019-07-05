#define _USE_MATH_DEFINES
#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <math.h>
#include <fstream>
#include <numeric>
#include <map>
#include <tuple>
#include <algorithm>
#include <random>
#include <set>
#include <functional>
#include <iomanip>
#include <ppl.h>
using namespace concurrency;

using namespace std;
using std::default_random_engine;
using std::uniform_int_distribution;
using std::cout;

using namespace Eigen;

typedef Eigen::SparseMatrix<double, StorageOptions::RowMajor> SpMat;// declares a column-major sparse matrix type of double
typedef SpMat SpMat_csr;
typedef Eigen::Triplet<double> T;
typedef Eigen::SparseVector<double> SpVec;

// 0	3	0	*	0
// 22   0	0	0	17
// 7	5	*	1	0
// 0	0	0	*	0
// 0	0	14	0	8


VectorXd zY(const vector<int>& ujList, const MatrixXd& Y, double* re = nullptr)
{
    assert(ujList.size() > 0);
    VectorXd z = VectorXd::Zero(Y.rows());
    for (auto& j: ujList)
    {
        z += Y.col(j);
    }
    double ru = 1.0 / sqrt(ujList.size());
    if (re)
    {
        *re = ru;
    }
    
    assert(Eigen::isinf(z.array()).any() == 0);
    return z * ru;
}

double predict(const RowVectorXd& Pu, const VectorXd& Qi, const RowVectorXd& z, double bu, double bi, double mu)
{
    return (Pu + z)*Qi + bu + bi + mu;
}

double costFunc(const vector<T>& tVec, const map<int, vector<int>>& uiMap, 
    const MatrixXd& P, const MatrixXd& Q, const VectorXd& bu, const VectorXd& bi, double mu, const MatrixXd& Y, double lambda)
{
    //double cst = 0.0;
    vector<double> diffVec(tVec.size(),0.0);
    //diffVec.reserve(tVec.size());
    parallel_transform(begin(tVec), end(tVec), diffVec.begin(), [&](auto& t) {
        int u = t.row();
        int i = t.col();
        double r = t.value() - predict(P.row(u), Q.col(i), zY(uiMap.at(u), Y).transpose(), bu(u), bi(i),mu);
        return r * r;
    });
    //double cost = accumulate(diffVec.begin(), diffVec.end(), 0.0);
    typedef vector<double>::iterator vd_iter;
    double cst = parallel_reduce(diffVec.begin(), diffVec.end(), 0.0, [](vd_iter block_begin, vd_iter block_end, double init)->double {
        return accumulate(block_begin, block_end, init);
    }, plus<double>());
    return cst/* + lambda * (bu.squaredNorm() + bi.squaredNorm() + P.squaredNorm() + Q.squaredNorm() + Y.squaredNorm())*/;
}

void learningSVDpp(const SpMat_csr& train_csr, MatrixXd *P, MatrixXd *Q, VectorXd* bu, VectorXd* bi, double mu, MatrixXd* Y
    ,int nEpoches, std::function<double(int)> stepFunc, double lambda)
{
    int m = train_csr.rows();
    int n = train_csr.cols();
    int F = (*Q).rows();

    map<int, vector<int>> uiMap;
    for (int u = 0; u < m; ++u)
    {
        int unnz = train_csr.outerIndexPtr()[u + 1] - train_csr.outerIndexPtr()[u];
        auto sptr = &train_csr.innerIndexPtr()[train_csr.outerIndexPtr()[u]];
        vector<int> ujList(sptr, sptr + unnz);
        uiMap[u] = std::move(ujList);
    }

    vector<T> tVec;
    tVec.reserve(train_csr.nonZeros());
    for (int k = 0; k < train_csr.outerSize(); ++k)
    {
        for (SpMat::InnerIterator it(train_csr, k); it; ++it)
        {
            tVec.emplace_back(it.row(), it.col(), it.value());
        }
    }

    typedef vector<double>::iterator vd_iter;
    cout << fixed << setprecision(5);
    cout << setfill(' ') << right;
    cout << "start iterations ..." << endl;
    vector<int> uVec(m);
    std::iota(begin(uVec), end(uVec), 0);
    vector<double> costVec;


    for (int epoch = 0; epoch < nEpoches; ++epoch)
    {
        double alpha = stepFunc(epoch);        
        shuffle(uVec.begin(), uVec.end(), std::default_random_engine(epoch));
        for (auto& u: uVec)
        {
            /*int unnz = train_csr.outerIndexPtr()[u + 1] - train_csr.outerIndexPtr()[u];
            auto sptr = &train_csr.innerIndexPtr()[train_csr.outerIndexPtr()[u]];
            vector<int> ujList(sptr, sptr + unnz);*/
            //assert(Eigen::isinf((*Y).array()).any() == 0);
            vector<int> & ujList = uiMap[u];
            if (ujList.empty())
            {
                continue;
            }
            double ru = 0.0;
            VectorXd z = zY(ujList, *Y, &ru);
            vector<double> eujList(ujList.size(), 0.0);
            VectorXd puz = (*P).row(u).transpose() + z;
            parallel_transform(ujList.begin(), ujList.end(), eujList.begin(), [&](auto& j) {
                return train_csr.coeff(u,j) - ((*Q).col(j).dot(puz) + (*bu)(u) + (*bi)(j) + mu);
            });
            /*auto eujListMaxVal = *max_element(eujList.begin(), eujList.end());
            auto eujListMinVal = *max_element(eujList.begin(), eujList.end());
            assert(eujListMaxVal < 1e100 && eujListMinVal > -1e100);*/

            //auto eujListSum = accumulate(eujList.begin(), eujList.end(), 0.0);
            double eujListSum = parallel_reduce(eujList.begin(), eujList.end(), 0.0
                , [](vd_iter block_begin, vd_iter block_end, double init)->double {
                return accumulate(block_begin, block_end, init);
            }, plus<double>());
            
            (*bu)(u) += alpha * (eujListSum - lambda * (*bu)(u));

            SpVec tmp(n)/* = VectorXd::Zero(n)*/;
            //tmp.reserve(500);
            //VectorXd tmp = VectorXd::Zero(n);
            //parallel_for (0, (int)(ujList.size()), [&](int k)
            //{
            //    tmp.coeffRef(ujList[k]) = eujList[k];
            //    //tmp(ujList[k]) = eujList[k];
            //});
            for (int k = 0; k < ujList.size(); ++k)
            {
                tmp.coeffRef(ujList[k]) = eujList[k];
            }


            //VectorXd dbiuj = -lambda * (*bi);
            //dbiuj += tmp;
            //(*bi) += alpha * dbiuj;


            //VectorXd Qeuj = VectorXd::Zero((*Q).rows());
            //vector<VectorXd> QeujVec(eujList.size());
            parallel_for(0, (int)(ujList.size()), [&ujList,&eujList, &bi, alpha, lambda/*, &Q,&QeujVec*/](int k) {
                (*bi)(ujList[k]) += alpha * (eujList[k] - lambda * (*bi)(ujList[k]));
                //QeujVec[k] = (*Q).col(ujList[k])*eujList[k];
                //tmp.coeffRef(ujList[k]) = eujList[k];
            });
            //VectorXd Qeuj = accumulate(begin(QeujVec), end(QeujVec),VectorXd::Zero(F)); // error
            /*VectorXd Qeuj = VectorXd::Zero(F);
            for (int k = 0; k < QeujVec.size(); ++k)
            {
                Qeuj += QeujVec[k];
            }*/

            VectorXd Qeuj = (*Q) * tmp;
            RowVectorXd dPu = Qeuj.transpose() - lambda * (*P).row(u);
            (*P).row(u) += alpha * dPu;

            VectorXd temp = ru * Qeuj;
            parallel_for(0, (int)(ujList.size()),[&](int k)
            {
                VectorXd dQuj = puz * eujList[k] - lambda * (*Q).col(ujList[k]);
                VectorXd dYuj = temp - lambda * (*Y).col(ujList[k]);

                (*Q).col(ujList[k]) += alpha * dQuj;
                (*Y).col(ujList[k]) += alpha * dYuj;
            });
            /*if (Eigen::isnan((*P).array()).any()
                || Eigen::isnan((*Q).array()).any()
                || Eigen::isnan((*bu).array()).any()
                || Eigen::isnan((*bi).array()).any()
                || Eigen::isnan((*Y).array()).any())
            {
                cout << "error" << endl;
            }*/
        }
        double cost = costFunc(tVec, uiMap, *P, *Q, *bu, *bi, mu, *Y, lambda);
        double penalty = lambda * ((*bu).squaredNorm() + (*bi).squaredNorm() + \
            (*P).squaredNorm() + (*Q).squaredNorm() + (*Y).squaredNorm());
        costVec.emplace_back(cost);
        cout << "Iteration:" << setw(4) << epoch \
            << ",    rmse:" << setw(12) << setprecision(9) << sqrt(cost / tVec.size())\
            << ",    mse:" << setw(12) << setprecision(5) << cost \
            << ",    penalty:" << setw(12) << penalty << endl;
    }
    cout.unsetf(ios::fixed);
}

void initModel(MatrixXd* P, MatrixXd* Q, VectorXd* bu, VectorXd* bi, MatrixXd* Y, int m, int n, int F = 100)
{
    /*default_random_engine e;
    e.seed(1);*/
    //uniform_real_distribution<double> u(0.0, 1.0); //随机数分布对象 
    srand(1/*(unsigned)time(NULL)*/);

    P->setRandom(m, F); 
    P->array() = (1 + P->array())*.5/sqrt(F);
    Q->setRandom(F, n);
    Q->array() = (1 + Q->array())*.5/sqrt(F);
    Y->setRandom(F, n);
    //Y->array() = (Y->array() + 1)*.5;
    bu->setZero(m);
    bi->setZero(n);
}

void setNegSample(SpMat& train_csr, int ratio = 1)
{
    train_csr.reserve(VectorXi::Constant(train_csr.rows(), 1000));
    default_random_engine e;
    e.seed(9); 
    uniform_real_distribution<double> u(0.0, 1.0); 
    double threshold = (1.0*ratio* train_csr.nonZeros()) / train_csr.size();
    parallel_for(0, (int)train_csr.rows(), [&](int i) {
        for (int j = 0; j < train_csr.cols(); ++j)
        {
            if (u(e) < threshold && (train_csr.coeff(i,j) == 0))
            {
                train_csr.coeffRef(i, j) = 0.0;
            }            
        }
    });  
    
    train_csr.makeCompressed();
}

int main(int argc, char** argv)
{
    int nn = Eigen::nbThreads();
    //srand(1/*(unsigned)time(NULL)*/);
    //Eigen::MatrixXd randvalue = (Eigen::MatrixXd::Random(4, 4)).array().abs() * 2 * M_PI;
    //std::cout << randvalue << std::endl;
    //cout << endl;
    //Eigen::MatrixXcd randvalue2 = Eigen::MatrixXcd::Random(4, 4);
    //std::cout << randvalue2 << std::endl;

    /*int rowList[] = { 1,2,2,3,0,2,4,2,1,4 };
    int colList[] = { 0,0,0,0,1,1,2,3,4,4 };
    double valList[] = { 22,7,2,0,3,5,14,1,17,8 };
    vector<T> coeVec(10);
    for (int i = 0; i < coeVec.size(); ++i)
    {
        coeVec[i] = T(rowList[i], colList[i], valList[i]);
    }
    SpMat BB(5,5);
    
    BB.setFromTriplets(coeVec.begin(), coeVec.end());
    cout << BB << endl;
    BB.reserve(VectorXi::Constant(5, 10));
    vector<T> extVec{ {0,3,0},{2,2,0},{3,3,0} };
    for (auto& v: extVec)
    {
        BB.coeffRef(v.row(), v.col()) += v.value();
    }
    cout << BB << endl;*/

    cout << "begin to read data ";
    int tmp1 = 0, tmp2 = 0;
    double tmp3 = 0.0;
    char ch;
    char buffer[256];
    ifstream input1(".\\train_data_8_tmp_train1");
    vector<T> coefficients;
    coefficients.reserve(2000000);
    //map<pair<int, int>, double> valMap;
    int maxRow = 0, maxCol = 0;
    while (input1.getline(buffer, 100))
    {
        stringstream ss(buffer);
        ss >> tmp1 >> ch >> tmp2 >> ch >> tmp3;
        coefficients.emplace_back(tmp1, tmp2, tmp3);
        /*rowVec.emplace_back(tmp1);
        colVec.emplace_back(tmp2);
        valVec.emplace_back(tmp3);*/
        //valMap[make_pair(tmp1, tmp2)] = tmp3;
        if(coefficients.size() % 10000 == 0) cout << '.';
        if (tmp1 > maxRow) maxRow = tmp1;
        if (tmp2 > maxCol) maxCol = tmp2;
    }
    input1.close();
    cout << "over" << endl;
    /*int rowList[] = { 1,2,2,3,0,2,4,2,1,4 };
    int colList[] = { 0,0,0,0,1,1,2,3,4,4 };
    double valList[] = { 22,7,2,0,3,5,14,1,17,8 };
    for (int i = 0; i < coefficients.size(); ++i)
    {
        coefficients[i] = T(rowList[i], colList[i], valList[i]);
    }*/

    auto mmv = minmax_element(coefficients.begin(), coefficients.end(), [](auto& a, auto& b) { return a.value() < b.value(); });
    cout << "rate value: min " << (mmv.first)->value() << ",     max " << mmv.second->value() << endl;

    cout << "start constructing train matrix ..." << endl;
    //auto maxKey = valMap.rbegin();
    int nRows = maxRow + 1;
    int nCols = maxCol + 1;
    SpMat A(nRows, nCols);
    /*A.reserve(VectorXi::Constant(m, 6));
    for (auto& v: coefficients)
    {
        A.coeffRef(v.row(), v.col()) += v.value();
    }*/


    A.setFromTriplets(coefficients.begin(), coefficients.end());
    /*A.coeffRef(0, 0) = 1.0;
    cout<< A.isCompressed() << endl;*/
    setNegSample(A, 2);
    cout << "train matrix finished." << endl;
    MatrixXd P, Q, Y;
    VectorXd bu, bi;
    cout << "start to init ..." << endl;
    initModel(&P, &Q, &bu, &bi, &Y, nRows, nCols);
    cout << "init finished." << endl;
    
    
    VectorXd coe = A.coeffs();
    double mu = coe.sum() / A.nonZeros();
    double t0 = 50.0, t1 = 0.06;
    auto learnSchedule = [t0,t1](int n) {return 1. / (t0 + n / t1); };
    learningSVDpp(A, &P, &Q, &bu, &bi, mu, &Y, 200, learnSchedule, 0.02);
   

    int array[8];
    for (int i = 0; i < 8; ++i) array[i] = i;
    cout << "Column-major:\n" << Map<Matrix<int, 2, 4> >(array) << endl;
    cout << "Row-major:\n" << Map<Matrix<int, 2, 4, RowMajor> >(array) << endl;
    cout << "Row-major using stride:\n" <<
        Map<Matrix<int, 2, 4>, Unaligned, Stride<1, 4> >(array) << endl;
    VectorXd z = -VectorXd::Ones(100);
    //z.swap(VectorXd(z.array().abs()));
    RowVectorXd y = z.array().abs();
    auto x = z.dot(y);
    double xx = y*z;
    cout << MatrixXd::Ones(5, 4) << endl;
    return 0;
}
