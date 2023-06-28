import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { HttpEventType, HttpResponse } from '@angular/common/http';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css']
})


export class DashboardComponent {

  file: File | null = null;
  predictionType = '';
  predictionPeriod = 0;
  predictedSales: number[] | undefined;
  actualSales: number[] | undefined;
  plotImage: string | undefined;

  constructor(private http: HttpClient) {}

  handleFileInput(event: Event): void {
    const target = event.target as HTMLInputElement;
    this.file = (target.files as FileList)[0];
  }

  onSubmit(): void {
    if (!this.file) {
      console.error('No file selected.');
      return;
    }

    const formData = new FormData();
    formData.append('file', this.file, this.file.name);
    formData.append('predictionType', this.predictionType);
    formData.append('predictionPeriod', this.predictionPeriod.toString());

    this.http.post<any>('http://localhost:5000/predict_sales', formData, {
      reportProgress: true,
      observe: 'events',
      responseType: 'blob' as 'json'
    }).subscribe(event => {
      if (event.type === HttpEventType.UploadProgress) {
        const percentDone = event.total ? Math.round(100 * (Number(event.loaded) / Number(event.total))) : 0;
        console.log(`File is ${percentDone}% uploaded.`);
      } else if (event instanceof HttpResponse) {
        console.log('File is completely uploaded!');
        console.log(event.body);

        const reader = new FileReader();
        reader.readAsDataURL(event.body);
        reader.onloadend = () => {
          this.plotImage = reader.result as string;
        };
      }
    }, error => {
      console.error(error);
    });
  }


}
