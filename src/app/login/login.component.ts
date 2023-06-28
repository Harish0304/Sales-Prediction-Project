import { Component, OnInit } from '@angular/core';
import { FormGroup , FormControl,Validators } from '@angular/forms';
import { Router } from '@angular/router';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit  {
  
  loginForm !: FormGroup ;
  hide:boolean=true;
  
  constructor(private router:Router) {
    this.loginForm = new FormGroup({
      username: new FormControl('', [Validators.required, Validators.email,Validators.pattern(
        'harish03042003@gmail.com',
      ),]),
      password: new FormControl('', [Validators.required,Validators.pattern(
        'Harish0304@'
      )])
    });
   }

  ngOnInit() :void{
    
  }
  onLogin()
  {
    if(!this.loginForm.valid){
      return;
    }
      this .router.navigateByUrl('/dashboard')
  }
  
}
