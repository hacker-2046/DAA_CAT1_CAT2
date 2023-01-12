#include <iostream>
#include <string>
using namespace std;
int maximum(int x,int y,int z)
{
    if (x >= y && x >= z)
        return x;
    else if (y >= x && y >= z)
        return y;
    else if (z >= x && z >= y)
        return z;
}
int main()
{
    int match,mismatch,max1,max2=1;
    string st1,st2;

    st1="acacacta";
    st2="agcacaca";

    match = 2;
    mismatch = -1;

    int m = st1.length();
    int n = st2.length();
    m+=1;
    n+=1;
    int scm[m][n];
    for (int i=0;i<m;i++)
    {
        scm[i][0] = 0;
        scm[0][i] = 0;
    }
    int max_score = 0;
    int max_i,max_j;
    for (int i=1;i<m;i++)
    {
        for (int j=1;j<n;j++)
        {
            if (st1[j-1] == st2[i-1])
            {
                scm[i][j] = scm[i-1][j-1] + match;
            }
            else
            {
                max1 = maximum(scm[i-1][j-1] + mismatch,scm[i-1][j] + mismatch,scm[i][j-1] + mismatch);
                scm[i][j] = max1;
            }
            if(scm[i][j] > max_score)
            {
                max_score = scm[i][j];
                max_i = i;
                max_j = j;
            }
        }
    }
    

    //printing the matrix with the strings string 1 in column and string 2 in row
    cout<<"---ALIGNMENT MATRIX---"<<endl;
    cout<<"  - ";
    for (int i=0;i<n;i++)
    {
        cout<<st1[i];
        cout<<" ";
    }
    cout<<"\n";
    for (int i=0;i<m;i++)
    {
        if (i==0)
            cout<<"- ";
        else
            cout<<st2[i-1]<<" ";
        for (int j=0;j<n;j++)
        {
            cout<<scm[i][j]<<" ";
        }
        cout<<"\n";
    }

    cout<<"---TRACEBACK---\n";
    int a=9,b=9;
    while (max2)
    {
    	max2 = maximum(scm[a-1][b-1],scm[a-1][b],scm[a][b-1]);
    	if (scm[a-1][b-1] == max2)
    	{
    		a -= 1;
    		b -= 1;
    		cout<<max2<<"->";
    	}
    	else if (scm[a-1][b] == max2)
    	{
    		a -= 1;
    		cout<<max2<<"->";
    	}
    	else if (scm[a][b-1] == max2)
    	{
    		b -= 1;
    		cout<<max2<<"->";
    	}
    }
    cout<<"\n";
    
    cout<<"---ALIGNMENT---"<<endl;
    int d = max_i;
    int e = max_j;
    
    while((d!=0 && e!=0)){
        if(st1[e-1] == st2[d-1]){
            cout << st1[e-1] << '\t' << st2[d-1] << endl;
            d -= 1;
            e -= 1;
        }     
        else{
            if(scm[d-1][e] > scm[d][e-1] && scm[d-1][e] > scm[d-1][e-1]){
                cout << '_' << '\t' << st2[d-1] << endl;
                d--;
            }     
            else if(scm[d][e-1] > scm[d-1][e] && scm[d][e-1] > scm[d-1][e-1]){
                cout << st1[e-1] << '\t' << '_' << endl;
                e--;
            }
        }
    }
}
